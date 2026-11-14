let isSoundEnabled = false;
let lastViolationCount = 0;
let seenViolations = new Set();

// Clock
function updateClock() {
    const now = new Date();
    document.getElementById('clock').innerText = now.toTimeString().split(' ')[0];
}
setInterval(updateClock, 1000);
updateClock();

// Tab Switcher
function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

    if (tabId === 'fileTab') {
        document.querySelectorAll('.tab-btn')[0].classList.add('active');
    } else {
        document.querySelectorAll('.tab-btn')[1].classList.add('active');
    }
    document.getElementById(tabId).classList.add('active');
}

// Sound Alarm Toggle (Default: OFF)
const soundBtn = document.getElementById('soundToggleBtn');
if (soundBtn) {
    soundBtn.classList.remove('active');
    document.getElementById('soundIcon').innerText = '🔇';
    soundBtn.innerHTML = '<span id="soundIcon">🔇</span> Səs Söndürülüb';
    soundBtn.addEventListener('click', function() {
        isSoundEnabled = !isSoundEnabled;
        this.classList.toggle('active', isSoundEnabled);
        this.innerHTML = isSoundEnabled ? '<span id="soundIcon">🔊</span> Səsli Siqnal Açıq' : '<span id="soundIcon">🔇</span> Səs Söndürülüb';
    });
}

function playAlertSound() {
    // Səs siqnalı istifadəçi tərəfindən söndürülüb
    if (!isSoundEnabled) return;
    const audio = document.getElementById('alertAudio');
    if (audio) {
        audio.currentTime = 0;
        audio.play().catch(e => {});
    }
}

// Video Select
function loadAvailableVideos() {
    fetch('/available_videos')
        .then(res => res.json())
        .then(videos => {
            const select = document.getElementById('videoSelect');
            select.innerHTML = '';
            videos.forEach(v => {
                const opt = document.createElement('option');
                opt.value = v;
                opt.textContent = v;
                select.appendChild(opt);
            });
        });
}

function changeSelectedVideo() {
    const videoName = document.getElementById('videoSelect').value;
    fetch('/set_source', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({type: 'file', video_name: videoName})
    }).then(res => res.json()).then(data => {
        document.getElementById('activeSourceDisplay').innerText = `Mənbə: ${videoName}`;
        reloadStream();
    });
}

// File Upload
const dropArea = document.getElementById('dropArea');
const fileInput = document.getElementById('videoFileInput');

dropArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropArea.style.borderColor = '#3b82f6';
});

dropArea.addEventListener('dragleave', () => {
    dropArea.style.borderColor = '#334155';
});

dropArea.addEventListener('drop', (e) => {
    e.preventDefault();
    dropArea.style.borderColor = '#334155';
    if (e.dataTransfer.files.length > 0) {
        handleUpload(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
        handleUpload(fileInput.files[0]);
    }
});

function handleUpload(file) {
    const formData = new FormData();
    formData.append('video', file);

    const progContainer = document.getElementById('uploadProgress');
    const progFill = document.getElementById('progressFill');
    const progText = document.getElementById('progressText');

    progContainer.style.display = 'flex';
    progFill.style.width = '0%';
    progText.innerText = 'Yüklənir: 0%';

    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/upload', true);

    xhr.upload.onprogress = function(e) {
        if (e.lengthComputable) {
            const percent = Math.round((e.loaded / e.total) * 100);
            progFill.style.width = percent + '%';
            progText.innerText = `Yüklənir: ${percent}%`;
        }
    };

    xhr.onload = function() {
        if (xhr.status === 200) {
            progText.innerText = 'Uğurla yükləndi!';
            progFill.style.backgroundColor = '#10b981';
            loadAvailableVideos();
            setTimeout(() => {
                progContainer.style.display = 'none';
                reloadStream();
            }, 1000);
        } else {
            progText.innerText = 'Xəta baş verdi!';
            progFill.style.backgroundColor = '#ef4444';
        }
    };

    xhr.send(formData);
}

// Camera Connect
function connectCamera() {
    const url = document.getElementById('rtspInput').value.trim();
    if (!url) {
        alert('Zəhmət olmasa RTSP və ya Kamera nömrəsi daxil edin');
        return;
    }
    fetch('/set_source', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({type: 'camera', url: url})
    }).then(res => res.json()).then(data => {
        document.getElementById('activeSourceDisplay').innerText = `Mənbə (Kamera): ${url}`;
        reloadStream();
    });
}

function setQuickRtsp(val) {
    document.getElementById('rtspInput').value = val;
}

// Confidence Slider
function updateConfidence(val) {
    document.getElementById('confVal').innerText = val + '%';
    fetch('/set_conf', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({confidence: val / 100.0})
    });
}

// Stream Control
function reloadStream() {
    const img = document.getElementById('videoStream');
    img.src = '/video_feed?t=' + new Date().getTime();
}

function handleStreamError() {
    console.log('Stream error, retrying...');
    setTimeout(reloadStream, 2000);
}

function toggleFullScreen() {
    const elem = document.querySelector('.stream-wrapper');
    if (!document.fullscreenElement) {
        elem.requestFullscreen().catch(err => alert(err.message));
    } else {
        document.exitFullscreen();
    }
}

// Live Stats Poller
setInterval(() => {
    fetch('/stats')
        .then(res => res.json())
        .then(data => {
            document.getElementById('kpiTotal').innerText = data.total_persons;
            document.getElementById('kpiSafe').innerText = data.hardhat_count;
            document.getElementById('kpiDanger').innerText = data.no_hardhat_count;
            const vestEl = document.getElementById('kpiVest');
            if (vestEl) vestEl.innerText = data.vest_count;
            document.getElementById('fpsDisplay').innerText = `FPS: ${data.fps}`;

            const dangerCard = document.getElementById('kpiDangerCard');
            const banner = document.getElementById('dangerBanner');

            if (data.no_hardhat_count > 0) {
                dangerCard.classList.add('danger-active');
                banner.style.display = 'block';
                playAlertSound();
            } else {
                dangerCard.classList.remove('danger-active');
                banner.style.display = 'none';
            }
        }).catch(e => console.log('Stats error:', e));
}, 500);

// Violations Poller
function pollViolations() {
    fetch('/violations')
        .then(res => res.json())
        .then(files => {
            const feed = document.getElementById('violationsFeed');
            if (files.length === 0) {
                feed.innerHTML = '<div class="empty-feed">Hələlik heç bir pozuntu qeydə alınmayıb.</div>';
                return;
            }

            if (files.length > seenViolations.size) {
                // Yeni pozuntu aşkarlandı
                feed.innerHTML = '';
                files.forEach(f => {
                    seenViolations.add(f.filename);
                    const card = document.createElement('div');
                    card.className = 'violation-card';
                    card.onclick = () => openModal(f.url, f.time);
                    card.innerHTML = `
                        <img src="${f.url}" class="violation-img" alt="Pozuntu şəkli" loading="lazy">
                        <div class="violation-meta">
                            <span class="violation-badge">⚠ KASKA YOXDUR</span>
                            <span class="violation-time">${f.time}</span>
                        </div>
                    `;
                    feed.appendChild(card);
                });
            }
        });
}
setInterval(pollViolations, 2000);
pollViolations();

// Clear Violations
function clearViolations() {
    if (confirm('Bütün pozuntu tarixçəsini silmək istəyirsiniz?')) {
        fetch('/clear_violations', {method: 'POST'})
            .then(res => res.json())
            .then(() => {
                seenViolations.clear();
                document.getElementById('violationsFeed').innerHTML = '<div class="empty-feed">Tarixçə təmizləndi.</div>';
            });
    }
}

// Modal View
function openModal(imgUrl, caption) {
    const modal = document.getElementById('imageModal');
    const modalImg = document.getElementById('modalImg');
    const modalCaption = document.getElementById('modalCaption');
    modal.style.display = 'flex';
    modalImg.src = imgUrl;
    modalCaption.innerText = `Xəbərdarlıq Snapshot: ${caption}`;
}

function closeModal() {
    document.getElementById('imageModal').style.display = 'none';
}

// Initial Call
loadAvailableVideos();
