This analysis provides insights into the Google Analytics 4 (GA4) performance of Vietnam Airlines (VNA), Vinpearl, and VinWonders over a three-day period from March 23-25, 2026.

---

### 1. Executive Summary

The overall performance across the three GA4 360 properties shows a mixed picture. **VNA** exhibits strong and stable traffic, predominantly from direct and organic sources, with a clear mobile-first audience. Direct revenue for VNA is positively trending, offsetting a decline in organic revenue. **Vinpearl** maintains stable traffic, led by Google Organic, and experienced a significant, positive anomaly with a massive spike in direct revenue on March 24th. **VinWonders**, however, presents a concerning trend with a consistent and significant decline across all key metrics – active users, page views, events, and particularly revenue – over the three-day period. A critical observation common to VNA and Vinpearl is the high volume of `(not set)` page paths, indicating a fundamental GA4 tracking configuration issue that limits content performance analysis.

---

### 2. Deep Dive into Each Property Group

#### VNA (Vietnam Airlines)

*   **Top Performers & Significant Patterns:**
    *   **(Direct) / (none)** is the leading traffic source, peaking at approximately 125,000 active users and showing a positive revenue trend, increasing from ~5.4e10 to ~5.8e10 VND (Vietnamese Dong) over the period. This indicates strong brand loyalty and direct booking success.
    *   **Mobile devices** are overwhelmingly dominant, consistently accounting for over 150,000 active users, underscoring the importance of a seamless mobile experience.
    *   **`page_view`** and **`user_engagement`** are the highest event counts, suggesting active browsing and interaction on the platform.
    *   Booking-related pages (`/booking/availability/0`, `/booking/shopping-cart`) are the most viewed specific content, indicating high purchase intent.
    *   **Concern:** While `google / organic` is the second-highest traffic source, its associated revenue is declining, from ~4e10 to ~3.2e10 VND.
    *   **Critical Issue:** The `(not set)` category represents the vast majority of page views, indicating a significant GA4 tracking issue that prevents detailed content performance analysis.

#### Vinpearl

*   **Traffic and Revenue Looking:**
    *   **Traffic:** `Google / organic` is the primary traffic driver, remaining stable around 27,000 active users. `(direct) / (none)` and `facebook / cpc` are consistent secondary sources. A positive trend is the significant surge in `facebook.com / referral` traffic on March 25th, nearly tripling, which could signal a successful social media initiative.
    *   **Revenue:** The most striking insight is a **massive spike in `(direct) / (none)` purchase revenue on March 24th**, exceeding 4e8 VND. This is a significant outlier and positive anomaly. Other revenue sources (Google Organic/CPC) show moderate fluctuations but are comparatively stable.
    *   **Hardware:** `Mobile` remains the dominant device, consistently around 35,000 active users, highlighting a mobile-centric audience.
    *   **Events:** `screen_view` is the highest event count and shows a slight increasing trend, suggesting strong app engagement or Single Page Application (SPA) interactions.
    *   **Critical Issue:** Similar to VNA, the `(not set)` category dominates page views, indicating a fundamental tracking flaw hindering content analysis.

#### VinWonders

*   **Specific Insights from Events and Hardware Reports:**
    *   **Events:** All major event counts (`user_engagement`, `page_view`, `session_start`) show a significant and concerning **downward trend** over the three days. `user_engagement` dropped from ~175,000 to ~110,000 events. This directly correlates with decreasing user activity and engagement.
    *   **Hardware:** Active users on both **`mobile` and `desktop` devices are declining steadily**, indicating a broad reduction in user reach and interest (e.g., mobile from ~42K to ~38K, desktop from ~17K to ~15K).
    *   **Traffic:** All primary traffic sources, including `google / organic` and `(direct) / (none)`, are in decline. `google / cpc` is relatively stable but not enough to offset the overall negative trend.
    *   **Revenue:** VinWonders faces a **severe and widespread revenue decline** across `(direct) / (none)`, `google / cpc`, and `google / organic` sources, particularly on March 24th. This is a major red flag indicating potential underlying issues impacting conversions.
    *   **Critical Issue:** The `(not set)` page path issue is also present, obscuring insights into which content might be underperforming or contributing to the overall decline.

---

### 3. Major Anomalies or Outliers

1.  **Widespread `(not set)` Page Paths (VNA, Vinpearl, VinWonders):** This is the most critical and pervasive anomaly. The dominance of `(not set)` for page views across VNA and Vinpearl, and also present in VinWonders, indicates a fundamental misconfiguration in GA4 tracking. This prevents any meaningful analysis of content performance, user navigation, and optimization efforts based on specific pages.
2.  **Vinpearl Direct Revenue Spike on March 24th:** The isolated and exceptionally high `(direct) / (none)` purchase revenue (over 4e8 VND) for Vinpearl on March 24th is a significant outlier. While positive, its singular nature requires immediate investigation to understand the cause (e.g., a high-value single transaction, a specific campaign's success, or a reporting anomaly) to inform future strategy or confirm data integrity.
3.  **VinWonders' Across-the-Board Decline:** The consistent and severe downward trend across all key performance indicators (active users, events, page views, and revenue) for VinWonders over just three days is a major anomaly requiring urgent attention. This rapid decline suggests a significant underlying problem, such as a major technical issue, a failed marketing campaign, or a critical change in market conditions.

---

### 4. Actionable Recommendations

1.  **Immediate GA4 Tracking Audit to Resolve `(not set)` Page Paths:**
    *   **Action:** Prioritize an urgent and comprehensive audit of the GA4 implementation for VNA, Vinpearl, and VinWonders to identify and rectify the root cause of `(not set)` page paths. Focus on ensuring `page_location` or `page_path` parameters are correctly populated via Google Tag Manager (GTM) or direct implementation.
    *   **Why:** Accurate page data is foundational for understanding user behavior, identifying high-performing content, diagnosing user journey drop-offs, and effectively optimizing the website/app experience. Without it, strategic decisions are made blindly.
    *   **Impact:** Enhanced data quality will enable more precise analysis of content performance, funnel optimization, and campaign effectiveness, directly contributing to improved user experience and conversion rates.

2.  **Launch a Comprehensive Investigation into VinWonders' Performance Collapse:**
    *   **Action:** Immediately initiate a deep-dive investigation into the rapid decline of traffic, engagement, and revenue for VinWonders. This should involve checking for recent website/app deployments, technical issues (e.g., site outages, broken tracking), changes in marketing campaigns (budgets, targeting), shifts in competitor activity, and cross-referencing with other available data sources (e.g., Google Search Console, CRM, internal sales data).
    *   **Why:** The severity and consistency of the decline suggest a critical issue that could be leading to substantial financial losses and brand damage. Understanding the cause is paramount to stopping the bleed and recovering performance.
    *   **Impact:** Swift identification and remediation of the root cause can prevent further revenue erosion, stabilize user base, and help restore overall business health for VinWonders.

3.  **Capitalize on Direct & Mobile Strengths, and Address VNA Organic Revenue Decay:**
    *   **Action for VNA:** Leverage the strong direct channel by investing further in loyalty programs, personalized direct marketing campaigns (e.g., email, app notifications), and exclusive direct booking incentives. Simultaneously, investigate the decline in `google / organic` revenue by conducting an SEO audit, analyzing keyword performance, checking landing page experience for organic users, and potentially updating content strategy.
    *   **Action for Vinpearl:** Deeply analyze the March 24th direct revenue spike to understand the contributing factors. If it was a replicable success (e.g., a specific offer, campaign, or customer segment), develop strategies to repeat or scale it. For both VNA and Vinpearl, continue optimizing the mobile user experience, given the dominance of mobile users.
    *   **Why:** Focusing on proven successful channels while addressing weaknesses ensures a balanced and growth-oriented strategy. Understanding specific success factors allows for replication and scaling.
    *   **Impact:** Increased direct revenue, halted organic revenue decline for VNA, and potentially scalable success for Vinpearl, leading to overall stronger financial performance and a more robust digital strategy.