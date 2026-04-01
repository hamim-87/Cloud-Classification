// ============================================================
//  Eyes on the Earth — CesiumJS 4-Point Satellite Imagery Selector
// ============================================================

(function () {
    'use strict';

    // =============================================
    //  Configuration
    // =============================================

    // NASA GIBS WMS endpoint (no API key required)
    const GIBS_WMS_BASE = 'https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi';

    // No Cesium Ion token needed — using free imagery providers

    const MAX_POINTS = 4;

    // =============================================
    //  State
    // =============================================
    let viewer = null;
    const selectedPoints = [];    // { lat, lon } decimal degrees
    const markerEntities = [];    // Cesium entity references
    let polygonEntity = null;
    let imageryOverlayLayer = null;
    let cloudLayer = null;

    // =============================================
    //  DOM References
    // =============================================
    const pointCountEl = document.getElementById('pointCount');
    const pointsListEl = document.getElementById('pointsList');
    const resetBtn = document.getElementById('resetBtn');
    const statusText = document.getElementById('statusText');
    const statusBadge = document.getElementById('statusBadge');
    const bboxPanel = document.getElementById('bboxPanel');
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingLayerEl = document.getElementById('loadingLayer');

    // =============================================
    //  1. initCesium()
    // =============================================
    function initCesium() {
        // Use OpenStreetMap tiles as base layer (free, no token, synchronous)
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

        // ---- Cloud layer: near-real-time GIBS MODIS/VIIRS corrected reflectance ----
        addCloudLayer();

        // Set default camera view
        viewer.camera.flyTo({
            destination: Cesium.Cartesian3.fromDegrees(0, 20, 20000000),
            duration: 0,
        });

        // Register click handler on the globe
        const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
        handler.setInputAction(function (click) {
            handleMapClick(click.position);
        }, Cesium.ScreenSpaceEventType.LEFT_CLICK);

        updateStatus('Ready — click on the globe');
        console.log('%c🌍 Eyes on the Earth — CesiumJS Initialized', 'font-size:14px;font-weight:bold;color:#4FC3F7');
    }

    // =============================================
    //  Cloud Layer — Real-time GIBS overlay
    // =============================================

    /**
     * Add a near-real-time cloud layer from NASA GIBS.
     * Uses VIIRS corrected reflectance (shows clouds naturally as white).
     * Overlaid semi-transparent so the base map shows through clear-sky areas.
     */
    function addCloudLayer() {
        const today = new Date();
        // GIBS typically has 1-3 day lag for corrected reflectance
        const dateStr = getDateString(today, -2);

        // GIBS WMTS tile URL for VIIRS Corrected Reflectance (True Color)
        // This layer shows the Earth with clouds as they appear naturally.
        // Using EPSG:3857 (GoogleMapsCompatible) tiling for web mercator.
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
        // Semi-transparent so we see base terrain + clouds blended on top
        cloudLayer.alpha = 0.55;
        cloudLayer.brightness = 1.15;

        console.log(`[Clouds] GIBS VIIRS cloud layer added for ${dateStr}`);
    }

    // =============================================
    //  2. handleMapClick(screenPosition)
    // =============================================
    function handleMapClick(screenPosition) {
        if (selectedPoints.length >= MAX_POINTS) {
            updateStatus('4 points already selected. Reset to start over.', 'error');
            return;
        }

        // Pick position on the globe
        const ray = viewer.camera.getPickRay(screenPosition);
        const cartesian = viewer.scene.globe.pick(ray, viewer.scene);

        if (!cartesian) {
            updateStatus('Click missed the globe — try again', 'error');
            return;
        }

        // Convert to geographic coordinates
        const cartographic = Cesium.Cartographic.fromCartesian(cartesian);
        const lat = Cesium.Math.toDegrees(cartographic.latitude);
        const lon = Cesium.Math.toDegrees(cartographic.longitude);

        // Store the point
        const point = { lat: parseFloat(lat.toFixed(6)), lon: parseFloat(lon.toFixed(6)) };
        selectedPoints.push(point);

        // Add visible marker on the globe
        addMarker(point, selectedPoints.length);

        // Update UI
        updatePointsUI();
        updateStatus(`Point ${selectedPoints.length} placed (${point.lat.toFixed(2)}°, ${point.lon.toFixed(2)}°)`);

        console.log(`[Point ${selectedPoints.length}] Lat: ${point.lat}, Lon: ${point.lon}`);

        // If we've collected all 4 points, proceed
        if (selectedPoints.length === MAX_POINTS) {
            updateStatus('4 points selected — computing bounding box...', 'success');
            const bbox = createBoundingBox(selectedPoints);
            displayBoundingBox(bbox);
            drawPolygon(selectedPoints);
            fetchSatelliteImage(bbox);
        }
    }

    // =============================================
    //  3. createBoundingBox(points)
    // =============================================
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

    // =============================================
    //  4. fetchSatelliteImage(bbox)
    // =============================================
    async function fetchSatelliteImage(bbox) {
        showLoading('Fetching satellite imagery...');

        const today = new Date();
        const layerId = 'VIIRS_SNPP_CorrectedReflectance_TrueColor';

        // Try multiple dates (GIBS has 1-3 day lag)
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
        } else {
            // Fallback: use ArcGIS World Imagery
            updateStatus('GIBS unavailable — using ArcGIS World Imagery', 'error');
            addArcGISOverlay(bbox);
        }

        // Fly to the selected area
        viewer.camera.flyTo({
            destination: Cesium.Rectangle.fromDegrees(
                bbox.west - 1, bbox.south - 1,
                bbox.east + 1, bbox.north + 1
            ),
            duration: 1.5,
        });

        hideLoading();
    }

    /**
     * Build a GIBS WMS GetMap URL for a single image covering the bbox.
     */
    function buildGIBSWmsUrl(layerId, dateStr, bbox, width = 1024, height = 1024) {
        // GIBS WMS 1.3.0 uses BBOX=minlat,minlon,maxlat,maxlon for EPSG:4326
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

    /**
     * Test if an image URL loads successfully (non-empty image).
     */
    function testImageLoad(url) {
        return new Promise((resolve) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => resolve(img.width > 0 && img.height > 0);
            img.onerror = () => resolve(false);
            img.src = url;
            // Timeout after 8 seconds
            setTimeout(() => resolve(false), 8000);
        });
    }

    /**
     * Overlay the fetched satellite image on the globe as a rectangle entity.
     */
    function addImageOverlay(imageUrl, bbox) {
        // Remove previous overlay entity
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

    /**
     * Fallback: add ESRI World Imagery as a tiled layer for the bbox.
     */
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
        // Store reference so reset can remove it
        imageryOverlayLayer = layer;
        // Mark as a layer (not entity) for proper cleanup
        imageryOverlayLayer._isImageryLayer = true;

        console.log('[Fallback] ESRI World Imagery layer added');
    }

    /**
     * Show a preview panel with the fetched satellite image + save & reset buttons.
     */
    function showImagePreview(imageUrl, dateStr, bbox) {
        // Remove existing preview if any
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

        // Save image button — downloads to user's default download folder
        document.getElementById('saveImageBtn').addEventListener('click', () => {
            saveImage(imageUrl, dateStr, bbox);
        });

        // Reset button in preview panel
        document.getElementById('resetFromPreviewBtn').addEventListener('click', () => {
            resetAll();
        });
    }

    /**
     * Save the satellite image by sending it to the FastAPI backend.
     * The backend stores it in the data/ folder.
     */
    const BACKEND_URL = 'http://localhost:8000';

    function saveImage(imageUrl, dateStr, bbox) {
        updateStatus('Saving image to server...', 'success');

        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = async () => {
            try {
                // Create canvas with satellite image + metadata footer
                const padding = 40;
                const canvas = document.createElement('canvas');
                canvas.width = img.width;
                canvas.height = img.height + padding;
                const ctx = canvas.getContext('2d');

                // Black background
                ctx.fillStyle = '#000';
                ctx.fillRect(0, 0, canvas.width, canvas.height);

                // Draw satellite image
                ctx.drawImage(img, 0, 0);

                // Metadata footer
                ctx.fillStyle = '#222';
                ctx.fillRect(0, img.height, canvas.width, padding);
                ctx.fillStyle = '#aaa';
                ctx.font = '14px Inter, sans-serif';
                ctx.fillText(
                    `NASA GIBS VIIRS | ${dateStr} | N:${bbox.north.toFixed(2)}° S:${bbox.south.toFixed(2)}° E:${bbox.east.toFixed(2)}° W:${bbox.west.toFixed(2)}°`,
                    10, img.height + 26
                );

                const filename = `satellite_${dateStr}_N${bbox.north.toFixed(2)}_S${bbox.south.toFixed(2)}_E${bbox.east.toFixed(2)}_W${bbox.west.toFixed(2)}.png`;

                // Convert canvas to Blob and send to backend
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
                // Fallback: download locally if backend is unreachable
                fallbackDownload(imageUrl, dateStr, bbox);
            }
        };
        img.onerror = () => {
            fallbackDownload(imageUrl, dateStr, bbox);
        };
        img.src = imageUrl;
    }

    /**
     * Fallback: trigger a browser download if the backend is not available.
     */
    function fallbackDownload(imageUrl, dateStr, bbox) {
        const link = document.createElement('a');
        const filename = `satellite_${dateStr}_N${bbox.north.toFixed(2)}_S${bbox.south.toFixed(2)}_E${bbox.east.toFixed(2)}_W${bbox.west.toFixed(2)}.png`;
        link.download = filename;
        link.href = imageUrl;
        link.target = '_blank';
        link.click();
        updateStatus('Image saved locally (backend unavailable)', 'success');
    }

    // =============================================
    //  5. drawPolygon(points)
    // =============================================
    function drawPolygon(points) {
        // Remove existing polygon
        if (polygonEntity) {
            viewer.entities.remove(polygonEntity);
            polygonEntity = null;
        }

        // Order points to form a proper polygon (convex hull ordering)
        const ordered = orderPointsForPolygon(points);

        // Flatten to [lon, lat, lon, lat, ...] for Cesium
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

        // Also draw the outline as a polyline for better visibility
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

    /**
     * Order points clockwise around their centroid so the polygon renders correctly.
     */
    function orderPointsForPolygon(points) {
        const cx = points.reduce((s, p) => s + p.lon, 0) / points.length;
        const cy = points.reduce((s, p) => s + p.lat, 0) / points.length;

        return [...points].sort((a, b) => {
            const angleA = Math.atan2(a.lat - cy, a.lon - cx);
            const angleB = Math.atan2(b.lat - cy, b.lon - cx);
            return angleA - angleB;
        });
    }

    // =============================================
    //  Marker Helpers
    // =============================================

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

    // =============================================
    //  UI Update Helpers
    // =============================================

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

    // =============================================
    //  Reset
    // =============================================

    function resetAll() {
        // Clear stored data
        selectedPoints.length = 0;
        markerEntities.length = 0;

        // Remove all entities (markers, polygon, outlines, rectangle overlay)
        viewer.entities.removeAll();

        // Remove imagery layer if it was an ArcGIS fallback layer
        if (imageryOverlayLayer && imageryOverlayLayer._isImageryLayer) {
            viewer.imageryLayers.remove(imageryOverlayLayer, true);
        }
        imageryOverlayLayer = null;

        // Remove image preview panel
        const preview = document.getElementById('imagePreview');
        if (preview) preview.remove();

        polygonEntity = null;

        // Reset UI
        updatePointsUI();
        bboxPanel.classList.add('hidden');
        updateStatus('Ready — click on the globe');

        // Reset camera
        viewer.camera.flyTo({
            destination: Cesium.Cartesian3.fromDegrees(0, 20, 20000000),
            duration: 1.0,
        });

        console.log('[Reset] All points and overlays cleared');
    }

    // =============================================
    //  Utility
    // =============================================

    function getDateString(date, offsetDays) {
        const d = new Date(date);
        d.setDate(d.getDate() + offsetDays);
        return d.toISOString().split('T')[0];
    }

    // =============================================
    //  Event Listeners
    // =============================================
    resetBtn.addEventListener('click', resetAll);

    // =============================================
    //  Bootstrap
    // =============================================
    initCesium();

    console.log('%c🌍 Eyes on the Earth — CesiumJS 4-Point Selector', 'font-size:14px;font-weight:bold;color:#4FC3F7');
    console.log('%c📡 NASA GIBS WMS endpoint ready', 'color:#81C784');
    console.log('%cClick 4 points on the globe to fetch satellite imagery', 'color:#aaa!!');
})();
