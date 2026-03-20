import cv2
import numpy as np
import urllib.request
import os
import json
import hashlib

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

GEOCOLOR_URL = "https://cdn.star.nesdis.noaa.gov/GOES16/ABI/FD/GEOCOLOR/latest.jpg"

BASE_OUTPUT_DIR = "satellite_results"


# -------------------------------------------------
# HELPERS
# -------------------------------------------------

def url_to_image(url: str):
    # Download an image from URL into an OpenCV BGR image.
    try:
        resp = urllib.request.urlopen(url, timeout=15)
        data = np.asarray(bytearray(resp.read()), dtype="uint8")
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None")
        return img
    except Exception as e:
        print(f"[ERROR] Failed to fetch image from {url}: {e}")
        return None


def resize_for_work(img, max_dim=1400):
    # Resize keeping aspect ratio so that max(width, height) = max_dim.
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1.0:
        return img
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def get_image_hash(img):
    # Generate a hash from the image to identify unique images.
    img_bytes = img.tobytes()
    return hashlib.md5(img_bytes).hexdigest()[:12]


def get_output_dir_for_image(img):
    # Get or create output directory for this specific image.
    img_hash = get_image_hash(img)
    output_dir = os.path.join(BASE_OUTPUT_DIR, img_hash)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# -------------------------------------------------
# EARTH DISK + GRID
# -------------------------------------------------

def find_earth_disk(img):
    # Approximate Earth disk center & radius from non-black region.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = img.shape[:2]
        r = min(h, w) // 2 - 10
        return (w // 2, h // 2), r
    c = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(c)
    return (int(x), int(y),), int(radius)


def draw_latlon_grid(img, center, radius):
    # Draw a rough latitude/longitude grid on top of img.
    cx, cy = center
    grid = img.copy()
    grid_color = (80, 80, 80)

    # Longitude lines (radials)
    for k in range(0, 12):
        angle = 2 * np.pi * k / 12
        x1 = int(cx + radius * np.cos(angle))
        y1 = int(cy + radius * np.sin(angle))
        x2 = int(cx - radius * np.cos(angle))
        y2 = int(cy - radius * np.sin(angle))
        cv2.line(grid, (x1, y1), (x2, y2), grid_color, 1)

    # Latitude circles
    for frac in [0.3, 0.5, 0.7]:
        r = int(radius * frac)
        cv2.circle(grid, center, r, grid_color, 1)

    return grid


# -------------------------------------------------
# CLOUD SEGMENTATION
# -------------------------------------------------

def compute_cloud_masks(img):
    # Compute: thick clouds, mid-level clouds, cirrus, combined mask.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Thick: very bright, low saturation
    thick = cv2.inRange(hsv, (0, 0, 215), (180, 55, 255))

    # Mid: medium-high V, medium S
    mid = cv2.inRange(hsv, (0, 20, 175), (180, 130, 235))

    # Cirrus/thin: tightened thresholds to reduce magenta
    cirrus = cv2.inRange(hsv, (0, 0, 160), (180, 60, 200))

    kernel_big = np.ones((9, 9), np.uint8)
    kernel_small = np.ones((5, 5), np.uint8)

    def clean(mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_big)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        return mask

    thick = clean(thick)
    mid = clean(mid)
    cirrus = clean(cirrus)

    combined = cv2.bitwise_or(thick, mid)
    combined = cv2.bitwise_or(combined, cirrus)

    return thick, mid, cirrus, combined


def cloud_classification_overlay(img, thick, mid, cirrus):
    # Color overlay: Thick=red, Mid=yellow, Cirrus=magenta.
    h, w = img.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    overlay[thick > 0] = (0, 0, 255)      # red
    overlay[mid > 0] = (0, 255, 255)      # yellow
    overlay[cirrus > 0] = (255, 0, 255)   # magenta

    return cv2.addWeighted(img, 0.6, overlay, 0.4, 0)


# -------------------------------------------------
# LAND / OCEAN + CLOUD STATS
# -------------------------------------------------

def segment_land_ocean(img):
    # Rough land vs ocean segmentation based on color.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ocean_mask = cv2.inRange(hsv, (80, 20, 20), (140, 255, 210))   # bluish
    land_mask = cv2.inRange(hsv, (20, 20, 30), (80, 255, 240))     # green/brown

    kernel = np.ones((5, 5), np.uint8)
    ocean_mask = cv2.morphologyEx(ocean_mask, cv2.MORPH_CLOSE, kernel)
    land_mask = cv2.morphologyEx(land_mask, cv2.MORPH_CLOSE, kernel)

    return land_mask, ocean_mask


def compute_cloud_stats(combined_cloud_mask, land_mask, ocean_mask):
    # Compute global, land, and ocean cloud cover percentages.
    total_pixels = combined_cloud_mask.size
    cloud_pixels = cv2.countNonZero(combined_cloud_mask)
    global_cover_pct = 100.0 * cloud_pixels / total_pixels

    land_pixels = cv2.countNonZero(land_mask)
    ocean_pixels = cv2.countNonZero(ocean_mask)

    land_cloud_mask = cv2.bitwise_and(combined_cloud_mask, combined_cloud_mask, mask=land_mask)
    ocean_cloud_mask = cv2.bitwise_and(combined_cloud_mask, combined_cloud_mask, mask=ocean_mask)

    land_cloud_pixels = cv2.countNonZero(land_cloud_mask)
    ocean_cloud_pixels = cv2.countNonZero(ocean_cloud_mask)

    land_cover_pct = 100.0 * land_cloud_pixels / land_pixels if land_pixels > 0 else 0.0
    ocean_cover_pct = 100.0 * ocean_cloud_pixels / ocean_pixels if ocean_pixels > 0 else 0.0

    stats = {
        "global_cloud_cover_pct": global_cover_pct,
        "land_cloud_cover_pct": land_cover_pct,
        "ocean_cloud_cover_pct": ocean_cover_pct,
    }
    return stats, land_cloud_mask, ocean_cloud_mask


# -------------------------------------------------
# STORM-LIKE CLUSTERS (LABELED)
# -------------------------------------------------

def detect_storm_clusters(thick_cloud_mask):
    # Detect large thick-cloud clusters as 'storms' and label them.
    contours, _ = cv2.findContours(thick_cloud_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    storms = []
    h, w = thick_cloud_mask.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    storm_index = 1
    font = cv2.FONT_HERSHEY_SIMPLEX

    for c in contours:
        area = cv2.contourArea(c)
        if area < 2000:
            continue

        storms.append((c, area))

        # Draw contour
        cv2.drawContours(overlay, [c], -1, (0, 0, 255), 2)

        # Label inside contour
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(overlay, f"Storm {storm_index}",
                        (cx - 30, cy),
                        font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            storm_index += 1

    storm_areas = [a for (_, a) in storms]
    return overlay, storm_areas


# -------------------------------------------------
# COASTLINES (SUBTLE)
# -------------------------------------------------

def extract_coastline_edges(img, combined_cloud_mask):
    # Approximate coastlines/borders using edges outside cloud regions.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

    # Remove edges inside clouds so we don't outline cloud boundaries.
    mask_inv = cv2.bitwise_not(combined_cloud_mask)
    edges = cv2.bitwise_and(edges, edges, mask=mask_inv)

    return edges


def overlay_coastlines(base, coast_edges):
    # Draw coastlines with light gray, slightly higher opacity.
    base = base.copy()
    coast_rgb = np.zeros_like(base)
    coast_rgb[coast_edges > 0] = (200, 200, 200)  # slightly brighter gray

    # Opacity 0.45 so they are visible but not overwhelming
    blended = cv2.addWeighted(base, 1.0, coast_rgb, 0.45, 0)
    return blended


# -------------------------------------------------
# SMOOTH CLOUD INTENSITY HEATMAP
# -------------------------------------------------

def build_intensity_heatmap(combined_cloud_mask):
    # Make a smooth density map from the cloud mask.
    cloud = combined_cloud_mask.astype(np.float32) / 255.0
    cloud = cv2.GaussianBlur(cloud, (31, 31), 0)
    cloud_norm = cv2.normalize(cloud, None, 0, 255, cv2.NORM_MINMAX)
    cloud_u8 = cloud_norm.astype(np.uint8)
    heatmap = cv2.applyColorMap(cloud_u8, cv2.COLORMAP_JET)
    return heatmap


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():
    print("[INFO] Downloading GeoColor image...")
    img = url_to_image(GEOCOLOR_URL)
    if img is None:
        print("[FATAL] Could not download GeoColor image.")
        return

    img = resize_for_work(img, max_dim=1400)
    
    # Get output directory for this specific image
    OUTPUT_DIR = get_output_dir_for_image(img)
    img_hash = get_image_hash(img)
    print(f"[INFO] Using output directory: {OUTPUT_DIR} (image hash: {img_hash})")
    
    center, radius = find_earth_disk(img)

    # Cloud masks
    thick, mid, cirrus, combined = compute_cloud_masks(img)

    # Classification overlay
    classified = cloud_classification_overlay(img, thick, mid, cirrus)

    # Land/ocean + stats
    land_mask, ocean_mask = segment_land_ocean(img)
    stats, land_cloud_mask, ocean_cloud_mask = compute_cloud_stats(combined, land_mask, ocean_mask)

    # Storm clusters
    storm_overlay, storm_areas = detect_storm_clusters(thick)

    # Smooth heatmap
    smooth_heatmap = build_intensity_heatmap(combined)

    # Coastline edges
    coast_edges = extract_coastline_edges(img, combined)

    # Apply coastlines & grid to each panel
    vis_orig = overlay_coastlines(img, coast_edges)
    vis_class = overlay_coastlines(classified, coast_edges)
    vis_storm = overlay_coastlines(storm_overlay, coast_edges)
    vis_heat = overlay_coastlines(smooth_heatmap, coast_edges)

    vis_orig = draw_latlon_grid(vis_orig, center, radius)
    vis_class = draw_latlon_grid(vis_class, center, radius)
    vis_storm = draw_latlon_grid(vis_storm, center, radius)
    vis_heat = draw_latlon_grid(vis_heat, center, radius)

    # Label panels
    h, w = vis_orig.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(vis_orig, "Original GeoColor", (20, 40),
                font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis_orig,
                f"Global clouds: {stats['global_cloud_cover_pct']:.1f}%",
                (20, 80), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis_orig,
                f"Land clouds:   {stats['land_cloud_cover_pct']:.1f}%",
                (20, 110), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis_orig,
                f"Ocean clouds:  {stats['ocean_cloud_cover_pct']:.1f}%",
                (20, 140), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(vis_class, "Cloud Type Classification", (20, 40),
                font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis_class,
                "Legend: Thick=Red  Mid=Yellow  Cirrus=Magenta",
                (20, 80), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(vis_storm, "Storm-like Thick Cloud Clusters", (20, 40),
                font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    if storm_areas:
        avg_area = sum(storm_areas) / len(storm_areas)
        cv2.putText(vis_storm,
                    f"Detected storms: {len(storm_areas)}",
                    (20, 80), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis_storm,
                    f"Largest: {max(storm_areas):.0f}px  Avg: {avg_area:.0f}px",
                    (20, 110), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(vis_storm,
                    "No large thick-cloud clusters detected",
                    (20, 80), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(vis_heat, "Smooth Cloud Intensity Heatmap", (20, 40),
                font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # Build 2x2 dashboard
    vis_class = cv2.resize(vis_class, (w, h))
    vis_storm = cv2.resize(vis_storm, (w, h))
    vis_heat = cv2.resize(vis_heat, (w, h))

    top = cv2.hconcat([vis_orig, vis_class])
    bottom = cv2.hconcat([vis_storm, vis_heat])
    dashboard = cv2.vconcat([top, bottom])

    # Title bar
    title_h = 60
    dash_h, dash_w = dashboard.shape[:2]
    title_bar = np.zeros((title_h, dash_w, 3), dtype=np.uint8)

    title_text = "GOES-16 GeoColor Cloud Analysis"

    cv2.putText(title_bar, title_text, (20, 40),
                font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    full_dashboard = cv2.vconcat([title_bar, dashboard])

    # Save outputs
    cv2.imwrite(os.path.join(OUTPUT_DIR, "geocolor_original.png"), img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "cloud_thick.png"), thick)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "cloud_mid.png"), mid)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "cloud_cirrus.png"), cirrus)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "cloud_combined.png"), combined)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "land_cloud_mask.png"), land_cloud_mask)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ocean_cloud_mask.png"), ocean_cloud_mask)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "storm_overlay.png"), storm_overlay)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "cloud_heatmap.png"), smooth_heatmap)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "coastline.png"), coast_edges)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "dashboard.png"), full_dashboard)

    # Print stats
    print("\n[STATS]")
    print(f"Global cloud cover: {stats['global_cloud_cover_pct']:.2f}%")
    print(f"Land cloud cover:   {stats['land_cloud_cover_pct']:.2f}%")
    print(f"Ocean cloud cover:  {stats['ocean_cloud_cover_pct']:.2f}%")
    if storm_areas:
        avg_area = sum(storm_areas) / len(storm_areas)
        print(f"Detected {len(storm_areas)} storm-like clusters.")
        print(f"Largest storm area: {max(storm_areas):.1f} px")
        print(f"Average storm area: {avg_area:.1f} px")
    else:
        print("No large thick-cloud storms detected.")

    # Export statistics to JSON file
    export_stats = {
        "cloud_cover": {
            "global_pct": round(stats['global_cloud_cover_pct'], 2),
            "land_pct": round(stats['land_cloud_cover_pct'], 2),
            "ocean_pct": round(stats['ocean_cloud_cover_pct'], 2)
        },
        "storms": {
            "count": len(storm_areas),
            "largest_area_px": round(max(storm_areas), 1) if storm_areas else 0,
            "average_area_px": round(sum(storm_areas) / len(storm_areas), 1) if storm_areas else 0,
            "areas_px": [round(area, 1) for area in storm_areas]
        }
    }
    
    stats_file = os.path.join(OUTPUT_DIR, "statistics.json")
    with open(stats_file, 'w') as f:
        json.dump(export_stats, f, indent=2)
    print(f"\n[INFO] Statistics exported to {stats_file}")

    cv2.imshow("GOES-16 GeoColor Cloud Analysis (Final)", full_dashboard)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()