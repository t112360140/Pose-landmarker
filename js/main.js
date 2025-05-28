import {
    PoseLandmarker,
    FilesetResolver,
    DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.js";

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const statusElement = document.getElementById("status");
const cameraSelection = document.getElementById("cameraSelection");
const shoulderDistanceDisplay = document.getElementById("shoulderDistanceDisplay");


let poseLandmarker = undefined;
let runningMode = "VIDEO";
let lastVideoTime = -1;
let drawingUtils;
let currentStream; // 用於儲存當前的媒體串流

// 在頁面載入時初始化 Pose Landmarker
async function createPoseLandmarker() {
    statusElement.innerText = "正在載入姿勢模型...";
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `models/pose_landmarker_lite.task`,
            delegate: "GPU",
        },
        runningMode: runningMode,
        numPoses: 3, // 允許多人偵測
        minPoseDetectionConfidence:0.8,
        minPosePresenceConfidence:0.8,
        minTrackingConfidence:0.8,
    });
    statusElement.innerText = "姿勢模型載入完成。";
    setupCamera(); // 模型載入完成後設置攝影機
}

// 獲取所有可用的攝影機並填充下拉選單
async function setupCameraSelection() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');

        cameraSelection.innerHTML = ''; // 清空現有選項
        if (videoDevices.length === 0) {
            cameraSelection.innerHTML = '<option>未找到攝影機</option>';
            statusElement.innerText = "未檢測到任何攝影機。";
            return;
        }

        videoDevices.forEach(device => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.text = device.label || `攝影機 ${device.deviceId}`;
            cameraSelection.appendChild(option);
        });

        // 監聽下拉選單的變化
        cameraSelection.addEventListener('change', () => {
            const selectedDeviceId = cameraSelection.value;
            startWebcam(selectedDeviceId);
        });

        // 預設啟動第一個攝影機
        startWebcam(videoDevices[0].deviceId);

    } catch (error) {
        console.error("列舉媒體裝置錯誤:", error);
        statusElement.innerText = `無法存取攝影機: ${error.name}`;
    }
}

// 啟動或切換攝影機
async function startWebcam(deviceId) {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop()); // 停止之前的串流
    }

    const constraints = {
        video: {
            deviceId: deviceId ? {
                exact: deviceId
            } : undefined
        }
    };

    try {
        currentStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = currentStream;
        video.addEventListener("loadeddata", predictWebcam);
        statusElement.innerText = "攝影機已啟動。";
        drawingUtils = new DrawingUtils(canvasCtx);
    } catch (error) {
        console.error("啟動攝影機錯誤:", error);
        statusElement.innerText = `無法存取攝影機: ${error.name} - ${error.message}`;
    }
}


// 設置攝影機，等待模型載入完成後調用
async function setupCamera() {
    if (poseLandmarker) {
        await setupCameraSelection(); // 在模型載入完成後才設置攝影機選擇
    } else {
        statusElement.innerText = "姿勢模型尚未載入，請稍候...";
        // 也可以考慮在這裡設定一個計時器，重複嘗試
    }
}


// 預測姿勢函式
async function predictWebcam() {
    // 設置畫布尺寸與影片相同
    canvasElement.style.width = video.videoWidth + "px";
    canvasElement.style.height = video.videoHeight + "px";
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;

    // 只有當模型和影片都準備好時才進行預測
    if (poseLandmarker && video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        const detections = poseLandmarker.detectForVideo(video, performance.now());

        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

        let distanceOutputHTML = ""; // 用於累積顯示的HTML

        if (detections.landmarks && detections.landmarks.length > 0) {
            for (let i = 0; i < detections.landmarks.length; i++) {
                const landmark = detections.landmarks[i]; // 2D 畫布座標 (正規化)
                const worldLandmark = detections.worldLandmarks[i]; // 3D 世界座標 (公尺)

                // 繪製骨架
                drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS, {
                    color: "#00FF00",
                    lineWidth: 4
                });
                drawingUtils.drawLandmarks(landmark, {
                    color: "#FF0000",
                    lineWidth: 2
                });

                // 取得左右肩膀的關鍵點索引
                const LEFT_SHOULDER = 11;
                const RIGHT_SHOULDER = 12;

                // 確保關鍵點存在
                if (landmark[LEFT_SHOULDER] && landmark[RIGHT_SHOULDER] &&
                    worldLandmark[LEFT_SHOULDER] && worldLandmark[RIGHT_SHOULDER]) {

                    // 1. 計算畫面上像素距離 (來自 normalized landmarks)
                    const leftShoulderPx = {
                        x: landmark[LEFT_SHOULDER].x * canvasElement.width,
                        y: landmark[LEFT_SHOULDER].y * canvasElement.height
                    };
                    const rightShoulderPx = {
                        x: landmark[RIGHT_SHOULDER].x * canvasElement.width,
                        y: landmark[RIGHT_SHOULDER].y * canvasElement.height
                    };

                    const pixelDistance = Math.sqrt(
                        Math.pow(rightShoulderPx.x - leftShoulderPx.x, 2) +
                        Math.pow(rightShoulderPx.y - leftShoulderPx.y, 2)
                    );

                    // 2. 計算真實世界 3D 距離 (來自 worldLandmarks)
                    const leftShoulderWorld = worldLandmark[LEFT_SHOULDER];
                    const rightShoulderWorld = worldLandmark[RIGHT_SHOULDER];

                    const trueWorld3DDistance = Math.sqrt(
                        Math.pow(rightShoulderWorld.x - leftShoulderWorld.x, 2) +
                        Math.pow(rightShoulderWorld.y - leftShoulderWorld.y, 2) +
                        Math.pow(rightShoulderWorld.z - leftShoulderWorld.z, 2)
                    );

                    // 3. 計算投影世界距離 (公尺) - 忽略 Z 軸，只在 XY 平面計算距離
                    // 這是您想要的"映射在畫面上實際的距離(公尺)"的最佳近似值，
                    // 它代表了在正面視角下，肩膀在真實世界中的平面距離
                    const projectedWorldDistanceXY = Math.sqrt(
                        Math.pow(rightShoulderWorld.x - leftShoulderWorld.x, 2) +
                        Math.pow(rightShoulderWorld.y - leftShoulderWorld.y, 2)
                    );

                    distanceOutputHTML += `
                        <p><strong>人物 ${i + 1}:</strong></p>
                        <p>肩膀距離 (畫布像素): ${pixelDistance.toFixed(2)} px</p>
                        <p>肩膀距離 (真實 3D 世界, 公尺): ${trueWorld3DDistance.toFixed(2)} m</p>
                        <p>肩膀距離 (XY 平面投影, 公尺): ${projectedWorldDistanceXY.toFixed(2)} m</p>
                    `;
                } else {
                    distanceOutputHTML += `<p><strong>人物 ${i + 1}:</strong> 無法偵測到完整的肩膀關鍵點。</p>`;
                }
            }
        } else {
            distanceOutputHTML = "<p>未偵測到任何人。</p>";
        }
        shoulderDistanceDisplay.innerHTML = distanceOutputHTML; // 更新顯示區塊
        canvasCtx.restore();
    }

    // 持續請求下一幀動畫
    window.requestAnimationFrame(predictWebcam);
}

// 頁面載入完成後執行
window.onload = createPoseLandmarker;