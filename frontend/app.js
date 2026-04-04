(function () {
    'use strict';

    const GIBS_WMS_BASE = 'https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi';
    const MAX_POINTS = 4;

    const CLASS_COLORS = {
        Fish:   { r: 0.255, g: 0.412, b: 0.882 },
        Flower: { r: 1.000, g: 0.843, b: 0.000 },
        Gravel: { r: 0.180, g: 0.545, b: 0.341 },
        Sugar:  { r: 0.863, g: 0.078, b: 0.235 },
    };
    const OVERLAY_ALPHA = 0.45;
    let lastSatelliteImageUrl = null;

    let viewer = null;
    const selectedPoints = [];
    const markerEntities = [];
    let polygonEntity = null;
    let imageryOverlayLayer = null;
    let cloudLayer = null;
    const pointCountEl = document.getElementById('pointCount');
    const pointsListEl = document.getElementById('pointsList');
    const resetBtn = document.getElementById('resetBtn');
    const statusText = document.getElementById('statusText');
    const statusBadge = document.getElementById('statusBadge');
    const bboxPanel = document.getElementById('bboxPanel');
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingLayerEl = document.getElementById('loadingLayer');
    const resultsPanel = document.getElementById('resultsPanel');
    const resultsLoading = document.getElementById('resultsLoading');
    const resultsBody = document.getElementById('resultsBody');
    const resultsCloseBtn = document.getElementById('resultsCloseBtn');

    function initCesium() {
        const osmBase = new Cesium.UrlTemplateImageryProvider({
            url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
            maximumLevel: 19,
            credit: 'Map data © OpenStreetMap contributors',
        });

        viewer = new Cesium.Viewer('cesiumContainer', {
            baseLayerPicker: false,
            geocoder: false,
            homeButton: false,
            navigationHelpButton: false,
            sceneModePicker: false,
            timeline: false,
            animation: false,
            fullscreenButton: false,
            selectionIndicator: false,
            infoBox: false,
            baseLayer: new Cesium.ImageryLayer(osmBase),
        });

        viewer.scene.globe.enableLighting = false;
        viewer.scene.globe.showGroundAtmosphere = true;

        addCloudLayer();

        viewer.camera.flyTo({
            destination: Cesium.Cartesian3.fromDegrees(0, 20, 20000000),
            duration: 0,
        });

        const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
        handler.setInputAction(function (click) {
            handleMapClick(click.position);
        }, Cesium.ScreenSpaceEventType.LEFT_CLICK);

        updateStatus('Ready — click on the globe');
        console.log('%c🌍 Eyes on the Earth — CesiumJS Initialized', 'font-size:14px;font-weight:bold;color:#4FC3F7');
    }

    function addCloudLayer() {
        const today = new Date();
        const dateStr = getDateString(today, -2);

        const gibsCloudProvider = new Cesium.WebMapTileServiceImageryProvider({
            url: 'https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/wmts.cgi',
            layer: 'VIIRS_SNPP_CorrectedReflectance_TrueColor',
            style: 'default',
            format: 'image/jpeg',
            tileMatrixSetID: 'GoogleMapsCompatible_Level9',
            maximumLevel: 9,
            tileWidth: 256,
            tileHeight: 256,
            credit: 'NASA GIBS — Suomi NPP / VIIRS',
            clock: viewer.clock,
            times: new Cesium.TimeIntervalCollection([
                new Cesium.TimeInterval({
                    start: Cesium.JulianDate.fromIso8601(dateStr),
                    stop: Cesium.JulianDate.fromIso8601(
                        getDateString(today, 1)
                    ),
                }),
            ]),
        });

        cloudLayer = viewer.imageryLayers.addImageryProvider(gibsCloudProvider);
        cloudLayer.alpha = 0.55;
        cloudLayer.brightness = 1.15;

        console.log(`[Clouds] GIBS VIIRS cloud layer added for ${dateStr}`);
    }

    function handleMapClick(screenPosition) {
        if (selectedPoints.length >= MAX_POINTS) {
            updateStatus('4 points already selected. Reset to start over.', 'error');
            return;
        }

        const ray = viewer.camera.getPickRay(screenPosition);
        const cartesian = viewer.scene.globe.pick(ray, viewer.scene);

        if (!cartesian) {
            updateStatus('Click missed the globe — try again', 'error');
            return;
        }

        const cartographic = Cesium.Cartographic.fromCartesian(cartesian);
        const lat = Cesium.Math.toDegrees(cartographic.latitude);
        const lon = Cesium.Math.toDegrees(cartographic.longitude);

        const point = { lat: parseFloat(lat.toFixed(6)), lon: parseFloat(lon.toFixed(6)) };
        selectedPoints.push(point);

        addMarker(point, selectedPoints.length);

        updatePointsUI();
        updateStatus(`Point ${selectedPoints.length} placed (${point.lat.toFixed(2)}°, ${point.lon.toFixed(2)}°)`);

        console.log(`[Point ${selectedPoints.length}] Lat: ${point.lat}, Lon: ${point.lon}`);

        if (selectedPoints.length === MAX_POINTS) {
            updateStatus('4 points selected — computing bounding box...', 'success');
            const bbox = createBoundingBox(selectedPoints);
            displayBoundingBox(bbox);
            drawPolygon(selectedPoints);
            fetchSatelliteImage(bbox);
        }
    }

    function createBoundingBox(points) {
        const lats = points.map(p => p.lat);
        const lons = points.map(p => p.lon);

        const bbox = {
            south: Math.min(...lats),
            north: Math.max(...lats),
            west: Math.min(...lons),
            east: Math.max(...lons),
        };

        console.log('[BBox]', bbox);
        return bbox;
    }

    async function fetchSatelliteImage(bbox) {
        showLoading('Fetching satellite imagery...');

        const today = new Date();
        const layerId = 'VIIRS_SNPP_CorrectedReflectance_TrueColor';

        let imageUrl = null;
        let usedDate = null;

        for (let offset = -1; offset >= -8; offset--) {
            const dateStr = getDateString(today, offset);
            const url = buildGIBSWmsUrl(layerId, dateStr, bbox);

            updateStatus(`Trying date ${dateStr}...`, 'success');
            console.log(`[GIBS] Trying: ${dateStr}`);

            const loaded = await testImageLoad(url);
            if (loaded) {
                imageUrl = url;
                usedDate = dateStr;
                break;
            }
        }

        if (imageUrl) {
            addImageOverlay(imageUrl, bbox);
            showImagePreview(imageUrl, usedDate, bbox);
            updateStatus(`Satellite imagery loaded — ${usedDate}`, 'success');
            console.log(`[GIBS] Imagery loaded for ${usedDate}`);
            // Automatically send image to backend for weather prediction
            sendImageForPrediction(imageUrl);
        } else {
            updateStatus('GIBS unavailable — using ArcGIS World Imagery', 'error');
            addArcGISOverlay(bbox);
        }

        viewer.camera.flyTo({
            destination: Cesium.Rectangle.fromDegrees(
                bbox.west - 1, bbox.south - 1,
                bbox.east + 1, bbox.north + 1
            ),
            duration: 1.5,
        });

        hideLoading();
    }

    function buildGIBSWmsUrl(layerId, dateStr, bbox, width = 1024, height = 1024) {
        const params = new URLSearchParams({
            SERVICE: 'WMS',
            REQUEST: 'GetMap',
            VERSION: '1.3.0',
            LAYERS: layerId,
            CRS: 'EPSG:4326',
            BBOX: `${bbox.south},${bbox.west},${bbox.north},${bbox.east}`,
            WIDTH: String(width),
            HEIGHT: String(height),
            FORMAT: 'image/jpeg',
            TIME: dateStr,
        });
        return `${GIBS_WMS_BASE}?${params.toString()}`;
    }

    function testImageLoad(url) {
        return new Promise((resolve) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => resolve(img.width > 0 && img.height > 0);
            img.onerror = () => resolve(false);
            img.src = url;
            setTimeout(() => resolve(false), 8000);
        });
    }

    function addImageOverlay(imageUrl, bbox) {
        if (imageryOverlayLayer) {
            viewer.entities.remove(imageryOverlayLayer);
            imageryOverlayLayer = null;
        }

        imageryOverlayLayer = viewer.entities.add({
            name: 'Satellite Imagery Overlay',
            rectangle: {
                coordinates: Cesium.Rectangle.fromDegrees(bbox.west, bbox.south, bbox.east, bbox.north),
                material: new Cesium.ImageMaterialProperty({
                    image: imageUrl,
                    transparent: false,
                }),
                heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
            },
        });

        console.log('[Overlay] Image rectangle entity added');
    }

    function addArcGISOverlay(bbox) {
        if (imageryOverlayLayer) {
            viewer.entities.remove(imageryOverlayLayer);
            imageryOverlayLayer = null;
        }

        const esri = new Cesium.UrlTemplateImageryProvider({
            url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            maximumLevel: 19,
            credit: 'Esri World Imagery',
        });

        const layer = viewer.imageryLayers.addImageryProvider(esri);
        layer.alpha = 1.0;
        imageryOverlayLayer = layer;
        imageryOverlayLayer._isImageryLayer = true;

        console.log('[Fallback] ESRI World Imagery layer added');
    }

    function showImagePreview(imageUrl, dateStr, bbox) {
        let preview = document.getElementById('imagePreview');
        if (preview) preview.remove();

        preview = document.createElement('div');
        preview.id = 'imagePreview';
        preview.style.cssText = `
            position: fixed; bottom: 30px; left: 20px; z-index: 100;
            background: rgba(10,14,20,0.92); border: 1px solid rgba(79,195,247,0.3);
            border-radius: 10px; padding: 12px; width: 280px;
            backdrop-filter: blur(14px); box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        `;
        preview.innerHTML = `
            <div style="font-size:11px; font-weight:700; letter-spacing:2px; color:#4FC3F7; margin-bottom:8px;">SATELLITE IMAGE</div>
            <div style="font-size:11px; color:#aaa; margin-bottom:8px;">Date: ${dateStr} &nbsp;|&nbsp; VIIRS True Color</div>
            <img id="previewImg" src="${imageUrl}" crossorigin="anonymous" style="width:100%; border-radius:6px; border:1px solid rgba(255,255,255,0.1);" alt="Satellite imagery"/>
            <div style="font-size:10px; color:#666; margin-top:6px; margin-bottom:10px;">Source: NASA GIBS — Suomi NPP / VIIRS</div>
            <div style="display:flex; gap:8px;">
                <button id="saveImageBtn" style="
                    flex:1; padding:9px; background:rgba(79,195,247,0.15); border:1.5px solid #4FC3F7;
                    color:#4FC3F7; font-size:12px; font-weight:600; letter-spacing:1px; cursor:pointer;
                    border-radius:6px; transition:all 0.2s;
                ">💾 Save Image</button>
                <button id="resetFromPreviewBtn" style="
                    flex:1; padding:9px; background:rgba(255,107,53,0.1); border:1.5px solid rgba(255,107,53,0.6);
                    color:#FF6B35; font-size:12px; font-weight:600; letter-spacing:1px; cursor:pointer;
                    border-radius:6px; transition:all 0.2s;
                ">🔄 Reset</button>
            </div>
        `;
        document.body.appendChild(preview);

        document.getElementById('saveImageBtn').addEventListener('click', () => {
            saveImage(imageUrl, dateStr, bbox);
        });

        document.getElementById('resetFromPreviewBtn').addEventListener('click', () => {
            resetAll();
        });
    }

    const BACKEND_URL = window.location.hostname === 'localhost' 
        ? 'http://localhost:8000' 
        : `http://${window.location.hostname}:8000`;

    function saveImage(imageUrl, dateStr, bbox) {
        updateStatus('Saving image to server...', 'success');

        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = async () => {
            try {
                const padding = 40;
                const canvas = document.createElement('canvas');
                canvas.width = img.width;
                canvas.height = img.height + padding;
                const ctx = canvas.getContext('2d');

                ctx.fillStyle = '#000';
                ctx.fillRect(0, 0, canvas.width, canvas.height);

                ctx.drawImage(img, 0, 0);

                ctx.fillStyle = '#222';
                ctx.fillRect(0, img.height, canvas.width, padding);
                ctx.fillStyle = '#aaa';
                ctx.font = '14px Inter, sans-serif';
                ctx.fillText(
                    `NASA GIBS VIIRS | ${dateStr} | N:${bbox.north.toFixed(2)}° S:${bbox.south.toFixed(2)}° E:${bbox.east.toFixed(2)}° W:${bbox.west.toFixed(2)}°`,
                    10, img.height + 26
                );

                const filename = `satellite_${dateStr}_N${bbox.north.toFixed(2)}_S${bbox.south.toFixed(2)}_E${bbox.east.toFixed(2)}_W${bbox.west.toFixed(2)}.png`;

                const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
                const formData = new FormData();
                formData.append('image', blob, filename);
                formData.append('filename', filename);

                const response = await fetch(`${BACKEND_URL}/save-image`, {
                    method: 'POST',
                    body: formData,
                });

                if (!response.ok) throw new Error(`Server error: ${response.status}`);

                const result = await response.json();
                updateStatus(`✅ Image saved: ${result.filename}`, 'success');
                console.log('[Save] Image saved to server:', result);

            } catch (err) {
                console.error('[Save] Backend save failed, falling back to download:', err);
                fallbackDownload(imageUrl, dateStr, bbox);
            }
        };
        img.onerror = () => {
            fallbackDownload(imageUrl, dateStr, bbox);
        };
        img.src = imageUrl;
    }

    function fallbackDownload(imageUrl, dateStr, bbox) {
        const link = document.createElement('a');
        const filename = `satellite_${dateStr}_N${bbox.north.toFixed(2)}_S${bbox.south.toFixed(2)}_E${bbox.east.toFixed(2)}_W${bbox.west.toFixed(2)}.png`;
        link.download = filename;
        link.href = imageUrl;
        link.target = '_blank';
        link.click();
        updateStatus('Image saved locally (backend unavailable)', 'success');
    }

    function drawPolygon(points) {
        if (polygonEntity) {
            viewer.entities.remove(polygonEntity);
            polygonEntity = null;
        }

        const ordered = orderPointsForPolygon(points);

        const positions = [];
        ordered.forEach(p => {
            positions.push(p.lon, p.lat);
        });

        polygonEntity = viewer.entities.add({
            name: 'Selected Area',
            polygon: {
                hierarchy: Cesium.Cartesian3.fromDegreesArray(positions),
                material: Cesium.Color.fromCssColorString('#4FC3F7').withAlpha(0.25),
                outline: true,
                outlineColor: Cesium.Color.fromCssColorString('#4FC3F7'),
                outlineWidth: 3,
                height: 0,
                heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
            },
        });

        viewer.entities.add({
            name: 'Selected Area Outline',
            polyline: {
                positions: Cesium.Cartesian3.fromDegreesArray([...positions, ordered[0].lon, ordered[0].lat]),
                width: 3,
                material: new Cesium.PolylineGlowMaterialProperty({
                    glowPower: 0.3,
                    color: Cesium.Color.fromCssColorString('#4FC3F7'),
                }),
                clampToGround: true,
            },
        });

        console.log('[Polygon] Drawn with', ordered.length, 'vertices');
    }

    function orderPointsForPolygon(points) {
        const cx = points.reduce((s, p) => s + p.lon, 0) / points.length;
        const cy = points.reduce((s, p) => s + p.lat, 0) / points.length;

        return [...points].sort((a, b) => {
            const angleA = Math.atan2(a.lat - cy, a.lon - cx);
            const angleB = Math.atan2(b.lat - cy, b.lon - cx);
            return angleA - angleB;
        });
    }



    function addMarker(point, index) {
        const entity = viewer.entities.add({
            name: `Point ${index}`,
            position: Cesium.Cartesian3.fromDegrees(point.lon, point.lat),
            point: {
                pixelSize: 14,
                color: Cesium.Color.fromCssColorString('#4FC3F7'),
                outlineColor: Cesium.Color.WHITE,
                outlineWidth: 2,
                heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
                disableDepthTestDistance: Number.POSITIVE_INFINITY,
            },
            label: {
                text: `P${index}  (${point.lat.toFixed(2)}°, ${point.lon.toFixed(2)}°)`,
                font: '13px Inter, sans-serif',
                fillColor: Cesium.Color.WHITE,
                outlineColor: Cesium.Color.BLACK,
                outlineWidth: 3,
                style: Cesium.LabelStyle.FILL_AND_OUTLINE,
                verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
                pixelOffset: new Cesium.Cartesian2(0, -20),
                heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
                disableDepthTestDistance: Number.POSITIVE_INFINITY,
            },
        });

        markerEntities.push(entity);
    }


    function updatePointsUI() {
        pointCountEl.textContent = selectedPoints.length;

        // Render point list
        pointsListEl.innerHTML = '';
        selectedPoints.forEach((p, i) => {
            const div = document.createElement('div');
            div.className = 'point-item';
            div.innerHTML = `
                <span class="point-num">${i + 1}</span>
                <span>${p.lat.toFixed(4)}°, ${p.lon.toFixed(4)}°</span>
            `;
            pointsListEl.appendChild(div);
        });
    }

    function displayBoundingBox(bbox) {
        document.getElementById('bboxNorth').textContent = bbox.north.toFixed(4) + '°';
        document.getElementById('bboxSouth').textContent = bbox.south.toFixed(4) + '°';
        document.getElementById('bboxEast').textContent = bbox.east.toFixed(4) + '°';
        document.getElementById('bboxWest').textContent = bbox.west.toFixed(4) + '°';
        bboxPanel.classList.remove('hidden');
    }

    function updateStatus(text, type) {
        statusText.textContent = text;
        statusBadge.className = 'status-badge';
        if (type === 'error') statusBadge.classList.add('error');
        if (type === 'success') statusBadge.classList.add('success');
    }

    function showLoading(layerName) {
        loadingOverlay.classList.remove('hidden');
        if (loadingLayerEl) loadingLayerEl.textContent = layerName || '';
    }

    function hideLoading() {
        loadingOverlay.classList.add('hidden');
    }


    // ── Weather Prediction ───────────────────────────────────────────────
    function sendImageForPrediction(imageUrl) {
        lastSatelliteImageUrl = imageUrl;
        // Show the results panel with loading state
        resultsPanel.classList.remove('hidden');
        resultsLoading.classList.remove('hidden');
        resultsBody.innerHTML = '';
        updateStatus('Analyzing cloud patterns...', 'success');

        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = async () => {
            try {
                // Draw raw satellite image to canvas (no padding/metadata)
                const canvas = document.createElement('canvas');
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0);

                const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
                const formData = new FormData();
                formData.append('image', blob, 'satellite_capture.png');

                const response = await fetch(`${BACKEND_URL}/predict-upload`, {
                    method: 'POST',
                    body: formData,
                });

                if (!response.ok) {
                    const errText = await response.text();
                    throw new Error(`Server error ${response.status}: ${errText}`);
                }

                const result = await response.json();
                console.log('[Predict] Result:', result);
                resultsLoading.classList.add('hidden');
                displayResults(result);
                updateStatus('Weather analysis complete', 'success');

            } catch (err) {
                console.error('[Predict] Error:', err);
                resultsLoading.classList.add('hidden');
                resultsBody.innerHTML = `<div class="results-error">Analysis failed: ${err.message}</div>`;
                updateStatus('Weather analysis failed', 'error');
            }
        };
        img.onerror = () => {
            resultsLoading.classList.add('hidden');
            resultsBody.innerHTML = '<div class="results-error">Failed to load satellite image for analysis.</div>';
            updateStatus('Could not load image for analysis', 'error');
        };
        img.src = imageUrl;
    }

    function loadImage(src) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = src;
        });
    }

    async function buildOverlay(satelliteImg, classResults) {
        const canvas = document.createElement('canvas');
        canvas.width = satelliteImg.width;
        canvas.height = satelliteImg.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(satelliteImg, 0, 0);

        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const pixels = imageData.data;

        const classNames = ['Fish', 'Flower', 'Gravel', 'Sugar'];
        for (const name of classNames) {
            const info = classResults[name];
            if (!info || !info.present || !info.mask_base64) continue;

            const color = CLASS_COLORS[name];
            const maskImg = await loadImage('data:image/png;base64,' + info.mask_base64);

            const maskCanvas = document.createElement('canvas');
            maskCanvas.width = canvas.width;
            maskCanvas.height = canvas.height;
            const maskCtx = maskCanvas.getContext('2d');
            maskCtx.drawImage(maskImg, 0, 0, canvas.width, canvas.height);
            const maskData = maskCtx.getImageData(0, 0, canvas.width, canvas.height).data;

            for (let i = 0; i < pixels.length; i += 4) {
                if (maskData[i] > 127) {
                    pixels[i]     = Math.round(pixels[i]     * (1 - OVERLAY_ALPHA) + OVERLAY_ALPHA * color.r * 255);
                    pixels[i + 1] = Math.round(pixels[i + 1] * (1 - OVERLAY_ALPHA) + OVERLAY_ALPHA * color.g * 255);
                    pixels[i + 2] = Math.round(pixels[i + 2] * (1 - OVERLAY_ALPHA) + OVERLAY_ALPHA * color.b * 255);
                }
            }
        }

        ctx.putImageData(imageData, 0, 0);
        return canvas.toDataURL('image/png');
    }

    async function displayResults(data) {
        let html = '';

        // Detected cloud types summary
        const detected = data.classes_detected || [];
        if (detected.length > 0) {
            html += `<div class="detected-summary">Detected: <strong>${detected.join(', ')}</strong></div>`;
        } else {
            html += `<div class="detected-summary">No significant cloud patterns detected.</div>`;
        }

        // Build overlay image
        const results = data.results || {};
        if (lastSatelliteImageUrl) {
            try {
                const satImg = await loadImage(lastSatelliteImageUrl);
                const overlayDataUrl = await buildOverlay(satImg, results);

                const legendHtml = Object.entries(CLASS_COLORS).map(([name, c]) => {
                    const conf = results[name] ? (results[name].confidence * 100).toFixed(1) : '0.0';
                    const cssColor = `rgb(${Math.round(c.r*255)},${Math.round(c.g*255)},${Math.round(c.b*255)})`;
                    return `<span class="legend-item">
                        <span class="legend-swatch" style="background:${cssColor}"></span>
                        ${name}: ${conf}%
                    </span>`;
                }).join('');

                html += `
                    <div class="overlay-section">
                        <div class="overlay-title">SEGMENTATION OVERLAY</div>
                        <img class="overlay-image" src="${overlayDataUrl}" alt="Segmentation overlay"/>
                        <div class="overlay-legend">${legendHtml}</div>
                    </div>
                `;
            } catch (err) {
                console.error('[Overlay] Failed to build overlay:', err);
            }
        }

        // Per-class cards
        for (const [className, info] of Object.entries(results)) {
            const isDetected = info.present;
            html += `
                <div class="cloud-type-card ${isDetected ? 'detected' : ''}">
                    <div class="cloud-type-header">
                        <span class="cloud-type-name">${className}</span>
                        <span class="cloud-type-badge ${isDetected ? 'present' : 'absent'}">
                            ${isDetected ? 'Detected' : 'Not found'}
                        </span>
                    </div>
                    <div class="cloud-type-stats">
                        <span><span class="label">Confidence:</span> ${(info.confidence * 100).toFixed(1)}%</span>
                        <span><span class="label">Coverage:</span> ${info.coverage_percent}%</span>
                    </div>
                    <div class="coverage-bar">
                        <div class="coverage-bar-fill" style="width: ${Math.min(info.coverage_percent, 100)}%"></div>
                    </div>
                </div>
            `;
        }

        // Weather analysis
        if (data.weather_analysis) {
            html += `
                <div class="weather-analysis-section">
                    <div class="weather-analysis-title">WEATHER FORECAST</div>
                    <div class="weather-analysis-text">${data.weather_analysis.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')}</div>
                </div>
            `;
        }

        resultsBody.innerHTML = html;
    }

    resultsCloseBtn.addEventListener('click', () => {
        resultsPanel.classList.add('hidden');
    });

    function resetAll() {
        // Clear stored data
        selectedPoints.length = 0;
        markerEntities.length = 0;

        // Remove all entities (markers, polygon, outlines, rectangle overlay)
        viewer.entities.removeAll();

        if (imageryOverlayLayer && imageryOverlayLayer._isImageryLayer) {
            viewer.imageryLayers.remove(imageryOverlayLayer, true);
        }
        imageryOverlayLayer = null;

        const preview = document.getElementById('imagePreview');
        if (preview) preview.remove();

        polygonEntity = null;

        // Hide results panel
        resultsPanel.classList.add('hidden');
        resultsBody.innerHTML = '';

        updatePointsUI();
        bboxPanel.classList.add('hidden');
        updateStatus('Ready — click on the globe');

        viewer.camera.flyTo({
            destination: Cesium.Cartesian3.fromDegrees(0, 20, 20000000),
            duration: 1.0,
        });

        console.log('[Reset] All points and overlays cleared');
    }

    function getDateString(date, offsetDays) {
        const d = new Date(date);
        d.setDate(d.getDate() + offsetDays);
        return d.toISOString().split('T')[0];
    }

    resetBtn.addEventListener('click', resetAll);

    initCesium();

    console.log('%c🌍 Eyes on the Earth — CesiumJS 4-Point Selector', 'font-size:14px;font-weight:bold;color:#4FC3F7');
    console.log('%c📡 NASA GIBS WMS endpoint ready', 'color:#81C784');
    console.log('%cClick 4 points on the globe to fetch satellite imagery', 'color:#aaa!!');
})();
