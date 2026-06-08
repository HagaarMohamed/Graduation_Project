## Nearby Tourist Attraction Finder
**Graduation Project 2025 · Faculty of Artificial Intelligence**

---

## What This Notebook Does

You tell it where you are. It tells you what's worth visiting nearby — and shows it all on an interactive map.

Specifically: you drop in your **latitude and longitude**, and the notebook goes out to **OpenStreetMap** (the open-source world map), pulls back every historical site, museum, monument, mosque, temple, palace, and tourist attraction within **8 km** of you, cleans up the messy data, and hands you back a ranked list of real places sorted by how close they are — along with a color-coded map you can click around in.

---

## The Problem It Solves

OpenStreetMap is incredibly detailed, but it's also noisy. A raw query for "attractions near me" returns restaurants, banks, bus stops, random people's homes, half-finished excavation sites, and government offices — mixed in with the actual landmarks. This notebook's main job is **separating the signal from the noise**.

It does that through a 5-layer filter that throws out:
- Places that aren't tagged as tourist-relevant at all
- Known junk (storage areas, parking lots, "Home", etc.)
- Ambiguous Arabic names that are probably just street addresses
- Bare personal names with no touristic context
- A long blocklist of noise keywords in both English and Arabic (restaurants, banks, schools, clinics, government offices, and more)

What survives that filter is a clean, deduplicated list of things a visitor would actually want to see.

---

## What You Get Out of It

1. **A ranked table** printed in the notebook — every attraction listed by rank, distance in km, and a proximity score out of 100
2. **An interactive map** rendered right inside Google Colab — your location as a red pin, a dashed circle showing your search radius, and color-coded pins for each attraction:
   - 🟢 Green = under 1 km away
   - 🔵 Blue = 1–3 km away
   - 🟠 Orange = 3–8 km away

Click any pin and you see the place's name, exact distance, score, and coordinates.

---

## How to Use It

1. Open the notebook in **Google Colab**
2. Find the `Set Your Location` cell and change `USER_LAT` and `USER_LON` to your coordinates
   - The default is set to **Cairo, Egypt** (`30.0444, 31.2357`)
3. Run all cells — that's it

> ⚠️ You need to enter coordinates manually. Colab doesn't have GPS access.

---

## Requirements

The notebook installs everything it needs automatically:

```
geopy    — calculates accurate real-world distances between GPS coordinates
folium   — builds the interactive map
groq     — (imported for potential AI integration)
requests — handles the API calls to OpenStreetMap
```

---

## How It Handles Failures

The OpenStreetMap API can be slow or temporarily unavailable. The notebook handles this gracefully by:
- Trying **3 different servers** (Germany, Austria, Russia) in order
- Retrying each one **up to 3 times** with increasing wait times between attempts
- Telling you clearly what went wrong if all options fail, and what to try next

---

## Project Context

This is **Pipeline 4** of the AtharAI graduation project — the location-aware attraction discovery component. It's designed to run in Google Colab and serves as the data-fetching and ranking backbone for the broader AtharAI tourism assistant system.
