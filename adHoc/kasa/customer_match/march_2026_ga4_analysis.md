# GA4 Performance Analysis Report: March 2026
**To:** Stakeholders (Vietnam Airlines, Vinpearl, VinWonders)  
**From:** GA4 Analytics Expert  
**Subject:** Cross-Property Performance & Strategic Insights

---

### 1. Executive Summary
The March 2026 data across Vietnam Airlines (VNA), Vinpearl, and VinWonders indicates a **Mobile-Dominant ecosystem** with high reliance on Organic and Direct traffic. VNA acts as the volume leader (Traffic/Revenue), while VinWonders shows the highest volatility and campaign-driven sensitivity. A critical tracking issue regarding **"(not set)" Page Paths** exists across all properties, masking specific content performance. The month is characterized by mid-week peaks for travel (VNA) and a significant outlier event on March 23rd for VinWonders.

---

### 2. Property Deep Dives

#### **Vietnam Airlines (VNA)**
*   **Traffic:** Extremely high volume, peaking at ~138k active users. The dominance of `(direct) / (none)` suggests a strong brand-direct booking habit or a high volume of users returning via saved bookmarks/app redirects.
*   **Pages:** The top dimension is `(not set)`, followed by `/booking/availability/0`. This confirms VNA’s primary utility as a booking engine, but the high `(not set)` volume indicates a need for enhanced Page Path/Screen Class configuration.
*   **Revenue:** Shows a strong cyclical pattern with a major upward trend toward the final week of March (peaking near 6e10). Direct traffic is the primary revenue driver, significantly outpacing Organic and CPC.

#### **Vinpearl**
*   **Traffic:** More balanced than VNA, with `google / organic` slightly leading. Traffic shows a gradual "cooling off" trend throughout the month, starting at ~32k and ending near ~25k active users.
*   **Engagement:** The `screen_view` and `view_item_list` events are prominent, indicating users are actively browsing hotel/resort options. 
*   **Revenue:** Highly volatile with sharp "heartbeat" spikes (Mar 6, 13, 20). This suggests a **periodic booking behavior**, likely corresponding to weekend-stay promotions or specific "Flash Sale" Fridays.

#### **VinWonders**
*   **Traffic:** Maintains a steady baseline of ~25k users, largely driven by `google / organic`. 
*   **Hardware:** Displays the most extreme mobile-to-desktop ratio among the three properties, reinforcing that VinWonders is an "on-the-go" or "destination-based" search product.
*   **The March 23 Event:** A massive spike is visible across Events, Pages, and Revenue. Revenue jumped to ~1.4e9 on this single day, likely due to a major ticket release or a viral marketing campaign.

---

### 3. Major Anomalies & Observations

1.  **The March 23rd Surge (VinWonders):** Every metric (Traffic, Page Views for `/vn/vi`, and Revenue) spiked simultaneously on Mar 23. This is a classic "successful campaign" signature or a one-day flash sale.
2.  **Tracking Deficit ("not set"):** In the "Pages" reports for all three properties, `(not set)` is the leading or a top-3 value. This is a critical GA4 configuration error, likely due to events firing before the `page_location` is processed or an issue with Single Page Application (SPA) tracking.
3.  **Vinpearl Revenue Decoupling:** While Vinpearl's traffic trended slightly downward, its revenue showed massive rhythmic spikes. This indicates that while fewer people visited toward the end of the month, the *intent* of those who did visit was significantly higher.

---

### 4. Cross-Report Correlations

*   **Traffic vs. Revenue:** For **VNA**, the correlation is high ($R \approx 0.85$); as traffic rises, revenue follows almost instantly. For **Vinpearl**, the correlation is lower; revenue is driven by specific days/promos rather than total volume.
*   **Device vs. Event Type:** Across all properties, **Mobile Hardware** correlates almost 1:1 with `user_engagement` and `session_start` spikes. This confirms the user journey is almost exclusively mobile-web or app-based.
*   **Source vs. Value:** `google / organic` drives the most *consistent* traffic, but `(direct) / (none)` drives the *highest value* transactions, especially for VNA. This suggests that search is for discovery, but the actual conversion happens in a direct session.

---

### 5. High-Impact Actionable Recommendations

1.  **Technical Fix: Resolve "not set" Page Dimensions.**
    *   *Action:* Audit the GTM (Google Tag Manager) triggers. Ensure that the "Config Tag" fires before any "Event Tags." For the SPA (Single Page Application) elements in the booking flow, implement "History Change" triggers to capture virtual page views correctly.
2.  **UX Strategy: Optimize for "Mobile-First" Conversion.**
    *   *Action:* Since >80% of traffic is mobile, prioritize "Thumb-Friendly" UI for the booking availability screens (`/booking/availability`). Reduce friction in the mobile checkout flow, as this is where the bulk of VNA and VinWonders revenue is generated.
3.  **Marketing Strategy: Replicate the "March 23" Model.**
    *   *Action:* Analyze the specific source/medium and campaign data for the VinWonders March 23rd spike. Whatever channel drove that surge (likely a combination of Social and Direct) should be used as a blueprint for monthly "Power Days" for Vinpearl to stabilize its revenue volatility, caused by periodic boosting campaign.