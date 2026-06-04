// ==========================================================================
//  SMART MIRROR DASHBOARD FRONTEND LOGIC - REAL-TIME AGENT & CONFIG
// ==========================================================================

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const liveTimeEl = document.getElementById("live-time");
    const liveDateEl = document.getElementById("live-date");
    
    const weatherTempEl = document.getElementById("weather-temp");
    const weatherDescEl = document.getElementById("weather-desc");
    const weatherHumidityEl = document.getElementById("weather-humidity");
    
    const toggleCameraBtn = document.getElementById("toggle-camera-btn");
    const videoStreamEl = document.getElementById("video-stream");
    const videoPlaceholderEl = document.getElementById("video-placeholder");
    
    const systemStatusEl = document.getElementById("system-status");
    const recognitionConfidenceEl = document.getElementById("recognition-confidence");
    
    const greetingCard = document.querySelector(".greeting-card");
    const userGreetingEl = document.getElementById("user-greeting");
    
    const healthAlertCard = document.querySelector(".health-alert-card");
    const healthAlertEl = document.getElementById("health-alert");
    const healthIconEl = document.getElementById("health-icon");
    const alertLabelEl = document.getElementById("alert-label");
    
    const statTempEl = document.getElementById("stat-temp");
    const statHumidityEl = document.getElementById("stat-humidity");
    const statusIndicator = document.querySelector(".status-indicator");
    
    // Modal Elements
    const configModal = document.getElementById("config-modal");
    const openConfigBtn = document.getElementById("open-config-btn");
    const closeModalBtn = document.getElementById("close-modal-btn");
    const saveConfigBtn = document.getElementById("save-config-btn");
    const addProfileBtn = document.getElementById("add-user-btn");
    const configUsersList = document.getElementById("config-users-list");
    
    // New Profile Input Fields
    const newUsernameInput = document.getElementById("new-username");
    const newGreetingInput = document.getElementById("new-greeting");
    const newAlertInput = document.getElementById("new-alert");
    
    let localConfig = {};
    let isCameraActive = true;

    // ===================================
    //  TIME & DATA TICKER (Backup display)
    // ===================================
    function updateClock() {
        const now = new Date();
        // Time format: HH:MM:SS AM/PM
        let hours = now.getHours();
        const minutes = String(now.getMinutes()).padStart(2, '0');
        const seconds = String(now.getSeconds()).padStart(2, '0');
        const ampm = hours >= 12 ? 'PM' : 'AM';
        hours = hours % 12;
        hours = hours ? hours : 12; // 0 should be 12
        
        liveTimeEl.textContent = `${String(hours).padStart(2, '0')}:${minutes}:${seconds} ${ampm}`;
        
        // Date format: Thursday, Jun 04, 2026
        const options = { weekday: 'long', year: 'numeric', month: 'short', day: '2-digit' };
        liveDateEl.textContent = now.toLocaleDateString('en-US', options);
    }
    
    setInterval(updateClock, 1000);
    updateClock();

    // ===================================
    //  REAL-TIME STATUS POLLING API
    // ===================================
    async function pollStatus() {
        try {
            const response = await fetch('/api/status');
            if (!response.ok) throw new Error("API network error");
            const data = await response.json();
            
            // Update User & Recognition details
            if (data.user && data.user !== "Unknown") {
                systemStatusEl.textContent = `Recognized: ${data.user}`;
                statusIndicator.classList.add("status-active");
                greetingCard.classList.add("user-detected");
                recognitionConfidenceEl.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
            } else {
                systemStatusEl.textContent = "Scanning for face...";
                statusIndicator.classList.remove("status-active");
                greetingCard.classList.remove("user-detected");
                recognitionConfidenceEl.textContent = "Confidence: --%";
            }
            
            // Update Greeting and Alert
            userGreetingEl.textContent = data.greeting;
            healthAlertEl.textContent = data.alert;
            
            // Format alerts aesthetics
            if (data.user && data.user !== "Unknown") {
                healthAlertCard.classList.add("alert-active");
                alertLabelEl.textContent = "Health Alert Reminders";
                healthIconEl.className = "fa-solid fa-heart-circle-exclamation";
            } else {
                healthAlertCard.classList.remove("alert-active");
                alertLabelEl.textContent = "General Advisory";
                healthIconEl.className = "fa-solid fa-heart-pulse";
            }
            
            // Update Weather Elements
            if (data.weather && data.weather.temp !== "N/A") {
                weatherTempEl.textContent = data.weather.temp;
                weatherHumidityEl.textContent = `Humidity: ${data.weather.humidity}%`;
                weatherDescEl.textContent = data.weather.description;
                
                statTempEl.textContent = `${data.weather.temp}°C`;
                statHumidityEl.textContent = `${data.weather.humidity}%`;
            }
            
        } catch (error) {
            console.error("Failed to poll status:", error);
            systemStatusEl.textContent = "Mirror disconnected";
            statusIndicator.classList.remove("status-active");
        }
    }
    
    // Poll every 1 second
    const pollInterval = setInterval(pollStatus, 1000);

    // ===================================
    //  CAMERA LIVE FEED TOGGLE
    // ===================================
    toggleCameraBtn.addEventListener("click", () => {
        isCameraActive = !isCameraActive;
        
        if (isCameraActive) {
            videoStreamEl.src = "/video_feed";
            videoStreamEl.classList.remove("hidden");
            videoPlaceholderEl.classList.add("hidden");
            toggleCameraBtn.innerHTML = `<i class="fa-solid fa-eye-slash"></i> Hide Feed`;
        } else {
            // Stop streaming from backend by cutting source
            videoStreamEl.removeAttribute("src");
            videoStreamEl.classList.add("hidden");
            videoPlaceholderEl.classList.remove("hidden");
            toggleCameraBtn.innerHTML = `<i class="fa-solid fa-eye"></i> Show Feed`;
        }
    });

    // ===================================
    //  CONFIG MODAL LOGIC (CRUD operations)
    // ===================================
    
    // Load config from backend
    async function loadConfigFromServer() {
        try {
            const response = await fetch('/api/config');
            if (!response.ok) throw new Error("Could not load configuration");
            localConfig = await response.json();
            renderConfigRows();
        } catch (error) {
            console.error("Config fetch error:", error);
        }
    }
    
    // Render configuration list rows
    function renderConfigRows() {
        configUsersList.innerHTML = "";
        
        Object.keys(localConfig).forEach(username => {
            // Skip the default fallback mapping if desired, but we let them edit it
            const row = document.createElement("div");
            row.className = "user-config-row";
            row.innerHTML = `
                <div class="user-row-header">
                    <span class="user-row-name">${username}</span>
                    ${username !== 'default' && username !== 'Unknown' ? 
                        `<button class="user-delete-btn" data-user="${username}" title="Delete User Profile">
                            <i class="fa-solid fa-trash-can"></i>
                         </button>` : ''
                    }
                </div>
                <div class="row-inputs">
                    <div>
                        <label style="font-size: 0.75rem; color: var(--text-muted);">Greeting Message</label>
                        <input type="text" class="config-greeting-input" data-user="${username}" value="${localConfig[username].greeting || ''}">
                    </div>
                    <div>
                        <label style="font-size: 0.75rem; color: var(--text-muted);">Health Alert</label>
                        <input type="text" class="config-alert-input" data-user="${username}" value="${localConfig[username].alert || ''}">
                    </div>
                </div>
            `;
            configUsersList.appendChild(row);
        });
        
        // Attach listener for delete buttons
        document.querySelectorAll(".user-delete-btn").forEach(btn => {
            btn.addEventListener("click", (e) => {
                const userToDelete = btn.getAttribute("data-user");
                delete localConfig[userToDelete];
                renderConfigRows();
            });
        });
        
        // Attach change listeners to inputs to save state in local memory
        document.querySelectorAll(".config-greeting-input").forEach(input => {
            input.addEventListener("input", (e) => {
                const user = input.getAttribute("data-user");
                localConfig[user].greeting = input.value;
            });
        });
        
        document.querySelectorAll(".config-alert-input").forEach(input => {
            input.addEventListener("input", (e) => {
                const user = input.getAttribute("data-user");
                localConfig[user].alert = input.value;
            });
        });
    }
    
    // Open settings modal
    openConfigBtn.addEventListener("click", () => {
        loadConfigFromServer().then(() => {
            configModal.classList.remove("hidden");
        });
    });
    
    // Close modal
    closeModalBtn.addEventListener("click", () => {
        configModal.classList.add("hidden");
    });
    
    // Close modal when clicking outside content
    window.addEventListener("click", (e) => {
        if (e.target === configModal) {
            configModal.classList.add("hidden");
        }
    });
    
    // Add new user profile to local structure
    addProfileBtn.addEventListener("click", () => {
        const username = newUsernameInput.value.trim().toLowerCase();
        const greeting = newGreetingInput.value.trim();
        const alert = newAlertInput.value.trim();
        
        if (!username) {
            alert("Username cannot be empty.");
            return;
        }
        
        if (localConfig[username]) {
            alert("User profile already exists.");
            return;
        }
        
        localConfig[username] = {
            greeting: greeting || `Hello, ${username}!`,
            alert: alert || "Stay healthy!"
        };
        
        // Clear fields
        newUsernameInput.value = "";
        newGreetingInput.value = "";
        newAlertInput.value = "";
        
        renderConfigRows();
    });
    
    // Save configuration and push to server
    saveConfigBtn.addEventListener("click", async () => {
        try {
            const response = await fetch('/api/config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(localConfig)
            });
            
            const result = await response.json();
            if (response.ok && result.status === "success") {
                configModal.classList.add("hidden");
                // Immediately poll status to update mirror greeting/alert
                pollStatus();
            } else {
                alert(`Error saving config: ${result.message}`);
            }
        } catch (error) {
            console.error("Failed to save config:", error);
            alert("Failed to communicate with settings server.");
        }
    });
});
