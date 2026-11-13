# News source evaluation (WP4)

Status: written from the API documentation as I know it. NOT yet tested live
(the build sandbox has no internet). Run the demo and update the "Observed" column.

| Source | Role | History | Cost/key | Limits to verify | Observed |
| --- | --- | --- | --- | --- | --- |
| Google News RSS | Live feed (implemented) | Days only | Free, no key | Terms of use for automated reading; ~100 items/query; result drift | _fill in_ |
| GDELT DOC 2.0 | Backfill + second opinion (implemented) | About 3 months in artlist mode | Free, no key | Strict rate limit; titles only; uneven Indian coverage | _fill in_ |
| NSE/BSE corporate announcements | Highest-quality company events (NOT implemented) | Yes | Free | Endpoints need session cookies; terms may forbid automated access. Check terms first; use official feeds/PDF pages only if permitted | _fill in_ |
| Free-tier news API (e.g. NewsAPI-style) | Optional (NOT implemented) | Usually 1 month, delayed | Key needed | Free plans are often non-commercial and delayed; check terms | _fill in_ |

Choice: Google News RSS for live, GDELT for recent history, our SQLite archive for
everything after the first run. Exchange filings are the best next addition if terms allow.
