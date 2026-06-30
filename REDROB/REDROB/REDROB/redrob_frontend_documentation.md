# RedRob Command Center — Frontend Technical Reference

> **Version:** 1.0.0  |  **Last updated:** 2026-06-26  |  **Stack:** React 18 + Vite 6

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Tech Stack & Dependencies](#2-tech-stack--dependencies)
3. [Directory Structure](#3-directory-structure)
4. [Environment Variables](#4-environment-variables)
5. [Authentication Flow](#5-authentication-flow)
6. [State Management](#6-state-management)
7. [API Layer Architecture](#7-api-layer-architecture)
8. [All API Endpoints (with schemas)](#8-all-api-endpoints-with-schemas)
9. [Data Models](#9-data-models)
10. [Pages & Routes](#10-pages--routes)
11. [Component Tree](#11-component-tree)
12. [Filters & Query Parameters](#12-filters--query-parameters)
13. [Production Backend Integration Guide](#13-production-backend-integration-guide)
14. [Build & Deployment](#14-build--deployment)
15. [Design System Notes](#15-design-system-notes)

---

## 1. Project Overview

**RedRob Command Center** is an AI-powered talent intelligence dashboard for evaluating and shortlisting high-signal research candidates. It ingests candidate profiles through a multi-layer neural pipeline (L1–L7), scores them, performs integrity checks, and presents a ranked shortlist for human review and scheduling.

**Key capabilities:**
- Real-time ranked candidate list with live filtering
- Per-candidate evidence drawer (quick-view) and full dossier page
- Pipeline funnel visualization (L1–L7 evaluation stages)
- System health & execution event log monitoring
- Supabase-backed authentication with JWT token propagation to the REST API

---

## 2. Tech Stack & Dependencies

### Runtime Dependencies

| Package | Version | Purpose |
|---|---|---|
| `react` | ^18.3.1 | UI framework |
| `react-dom` | ^18.3.1 | DOM renderer |
| `@tanstack/react-query` | ^5.66.0 | Server-state caching, re-fetching |
| `zustand` | ^5.0.3 | Client-state management (global store) |
| `axios` | ^1.7.9 | HTTP client |
| `@supabase/supabase-js` | ^2.49.4 | Auth (Supabase JWT) |
| `framer-motion` | ^12.0.6 | Page transitions, drawer animations |
| `lucide-react` | ^0.468.0 | Icon set |
| `three` | ^0.172.0 | 3D WebGL rendering (Three.js) |
| `@react-three/fiber` | ^8.17.10 | React renderer for Three.js |
| `@react-three/drei` | ^9.122.0 | Three.js helpers & abstractions |
| `@react-spring/three` | ^9.7.5 | Spring physics animations in 3D scenes |
| `@tsparticles/react` | ^4.2.1 | Particle background (login page) |
| `@tsparticles/slim` | ^4.2.1 | Lightweight particle engine |

### Dev Dependencies

| Package | Version | Purpose |
|---|---|---|
| `vite` | ^6.1.0 | Build tool & dev server |
| `@vitejs/plugin-react` | ^4.3.4 | Babel/JSX transform for Vite |
| `tailwindcss` | ^3.4.17 | Utility CSS (available, minimally used) |

### Global QueryClient Configuration

```js
// src/main.jsx
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,   // No refetch on tab switch
      retry: 1,                      // One retry on failure
      staleTime: 30_000              // Data considered fresh for 30s
    }
  }
})
```

---

## 3. Directory Structure

```
redrob/
├── index.html                    # Vite HTML entry
├── vite.config.js                # Build config (code splitting)
├── tailwind.config.js            # Tailwind config
├── package.json
├── .env                          # Active env (git-ignored)
├── .env.example                  # Template for env vars
│
└── src/
    ├── main.jsx                  # App bootstrap, providers
    ├── App.jsx                   # Auth gate + dark mode controller
    ├── styles.css                # Global design system (CSS variables, all component styles)
    │
    ├── api/
    │   ├── client.js             # Axios instance (base URL, JWT interceptor, 401 handler)
    │   ├── redrobApi.js          # Public API functions (mock-or-real routing logic)
    │   ├── mockApi.js            # Local mock implementations (uses data.js)
    │   └── supabase.js           # Supabase client + signIn/signOut helpers
    │
    ├── hooks/
    │   ├── useCandidates.js      # useCandidates(), useCandidate(id)
    │   ├── usePipeline.js        # usePipelineRuns(), usePipelineFunnel(runId)
    │   └── useSystemHealth.js    # useSystemHealth(), useSystemEvents()
    │
    ├── store/
    │   └── useAppStore.js        # Zustand global store (all app state)
    │
    ├── mocks/
    │   └── data.js               # Static mock data (candidates, pipeline, health, events)
    │
    ├── pages/
    │   ├── LoginPage.jsx         # Auth screen with particle background
    │   ├── Overview.jsx          # Dashboard (KPIs, funnel, risk, shortlist preview)
    │   ├── RankedCandidates.jsx  # Main candidate list + filter rail + evidence drawer
    │   ├── CandidateDetail.jsx   # Full dossier view for one candidate
    │   └── SystemHealth.jsx      # System diagnostics + event log
    │
    └── components/
        ├── layout/
        │   ├── AppShell.jsx      # Root layout (sidebar + topbar + page outlet)
        │   ├── Sidebar.jsx       # Left navigation
        │   ├── TopBar.jsx        # Header bar (search, dark mode, auth)
        │   └── PageWrapper.jsx   # Animated page transition wrapper
        │
        ├── candidates/
        │   ├── CandidateTable.jsx    # Ranked candidate list rows
        │   ├── EvidenceDrawer.jsx    # Slide-in evidence panel
        │   └── FilterRail.jsx        # Left filter sidebar (score, domain, fraud)
        │
        ├── overview/
        │   ├── FunnelChart.jsx       # L1–L7 pipeline funnel bar visualization
        │   ├── KPIStrip.jsx          # KPI metric strip
        │   └── RiskCards.jsx         # System risk/health cards
        │
        ├── ui/
        │   ├── Badge.jsx             # Status badge pill
        │   ├── LogicMatrix.jsx       # Logic score grid display
        │   ├── ProgressBar.jsx       # Animated progress bar
        │   ├── ScrollShowcase.jsx    # Scroll-animated section wrapper
        │   ├── ShapeLandingHero.jsx  # 3D animated hero section
        │   └── Sparkles.jsx          # tsParticles wrapper
        │
        └── 3d/
            ├── CommandField3D.jsx    # Animated 3D command field scene
            ├── NavIcon3D.jsx         # 3D icon for navigation
            └── ScoreOrb3D.jsx        # Rotating 3D score orb
```

---

## 4. Environment Variables

All env vars are prefixed with `VITE_` (exposed to the browser by Vite).

```env
# .env (copy from .env.example)

# Base URL for your REST API backend
VITE_API_URL=http://localhost:8000

# Toggle mock API — set to 'false' to hit real backend
VITE_USE_MOCK_API=true

# Supabase project URL
VITE_SUPABASE_URL=https://xxxx.supabase.co

# Supabase public anon key
VITE_SUPABASE_ANON_KEY=replace-with-supabase-anon-key
```

> [!IMPORTANT]
> Setting `VITE_USE_MOCK_API=false` switches **all** API calls to hit your real backend at `VITE_API_URL`. The fallback-to-mock behavior is only active in try/catch for non-401 errors.

> [!NOTE]
> If `VITE_SUPABASE_URL` contains `xxxx` or `VITE_SUPABASE_ANON_KEY` contains `replace-with`, the Supabase client is **not initialized** and the login page shows "Local Preview" mode (accepts any email/password).

---

## 5. Authentication Flow

### Login Sequence

```
User submits form
    → signIn(email, password)   [src/api/supabase.js]
        → supabase.auth.signInWithPassword()  (if Supabase configured)
        → OR: returns local-preview token      (if no Supabase config)
    → login({ token, user })    [useAppStore]
        → stores token in localStorage as 'supabase_token'
        → sets authToken in Zustand state
    → App renders <AppShell />  (auth gate: authToken is truthy)
```

### JWT Propagation

Every API request automatically includes the JWT from `localStorage`:

```js
// src/api/client.js — request interceptor
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('supabase_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})
```

### 401 Handling

```js
// src/api/client.js — response interceptor
if (error.response?.status === 401) {
  window.dispatchEvent(new CustomEvent('redrob:unauthorized'))
}
```

```js
// src/App.jsx — listens globally
window.addEventListener('redrob:unauthorized', () => {
  signOut()   // Supabase session invalidation
  logout()    // Clears localStorage + Zustand state → shows LoginPage
})
```

### Logout

`useAppStore.logout()` clears:
- `localStorage.removeItem('supabase_token')`
- Zustand: `{ authToken: null, user: null, activePage: 'overview', drawerOpen: false, activeCandidateId: null }`

---

## 6. State Management

**Library:** Zustand v5 (single global store)  
**File:** `src/store/useAppStore.js`

### State Shape

```ts
{
  // Navigation
  activePage: 'overview' | 'candidates' | 'detail' | 'health'

  // Auth
  authToken: string | null      // JWT from Supabase or local preview
  user: object | null           // Supabase user object

  // UI Preferences
  darkMode: boolean             // Persisted to localStorage as 'redrob_dark_mode'

  // Candidate selection (Evidence Drawer)
  activeCandidateId: string | null   // e.g. 'cand-alexei-kozlov'
  drawerOpen: boolean                // Controls EvidenceDrawer visibility

  // Candidate list filters
  filters: {
    score_min: number            // Default: 70
    verified_only: boolean       // Default: true
    domain: string               // Default: 'all'
    page: number                 // Default: 1
    limit: number                // Default: 20
  }
}
```

### Actions

| Action | Effect |
|---|---|
| `hydrate()` | Reads localStorage, applies dark mode class to `<html>` |
| `setActivePage(page)` | Changes visible page |
| `toggleDark()` | Flips dark mode, persists to localStorage |
| `login({ token, user })` | Persists token, sets auth state |
| `logout()` | Clears all auth state and localStorage |
| `setActiveCandidate(id)` | Sets `activeCandidateId` **and** `drawerOpen: true` |
| `closeDrawer()` | Sets `drawerOpen: false` and `activeCandidateId: null` |
| `setFilters(partial)` | Merges partial filter, resets `page` to 1 |
| `resetFilters()` | Restores default filter values |

> [!NOTE]
> `setActiveCandidate()` is the **only** way to open the Evidence Drawer. It must be called by a user interaction (clicking a candidate row or overview table row). The drawer never opens automatically.

---

## 7. API Layer Architecture

The API layer has a **mock-or-real** routing mechanism:

```
redrobApi.js
    └── remoteOrMock(remoteCall, mockCall)
            ├── if VITE_USE_MOCK_API === true   → always runs mockCall()
            └── if VITE_USE_MOCK_API === false
                    ├── tries remoteCall() via axios
                    ├── on 401 → throws (triggers global logout)
                    └── on any other error → falls back to mockCall()
```

**Axios instance** (`src/api/client.js`):
- `baseURL`: `VITE_API_URL` (default `http://localhost:8000`)
- `timeout`: 12,000 ms
- Automatic `Authorization: Bearer <token>` header

---

## 8. All API Endpoints (with schemas)

The frontend makes **7 API calls** across 5 resource groups.

---

### 8.1 `GET /candidates`

**Used by:** `useCandidates()` hook → `RankedCandidates` page, `Overview` page  
**Trigger:** On mount + whenever filter state changes (React Query key: `['candidates', filters]`)

#### Query Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `score_min` | `number` | `70` | Minimum composite score (0–100) |
| `verified_only` | `boolean` | `true` | Only return `integrity_status === 'CLEAN'` candidates |
| `domain` | `string` | `'all'` | Filter by domain: `'all'`, `'core-ml'`, `'systems'`, `'alignment'` |
| `page` | `number` | `1` | Pagination page number |
| `limit` | `number` | `20` | Results per page |

#### Example Request
```
GET /candidates?score_min=70&verified_only=true&domain=all&page=1&limit=20
Authorization: Bearer <jwt>
```

#### Expected Response — `200 OK`
```json
[
  {
    "id": "cand-alexei-kozlov",
    "rank": "01",
    "initials": "AK",
    "name": "Alexei Kozlov",
    "role": "Principal ML Systems Researcher",
    "background": "PhD, Stanford - 8 years experience",
    "location": "San Francisco, CA",
    "domain": "core-ml",
    "score": 94,
    "match": "98.4%",
    "integrity_status": "CLEAN",
    "dominant_trait": "Core Algorithm",
    "verified_at": "2026-06-26T04:40:00Z",
    "logic_scores": { "l2": 94, "l3": 88, "l4": 92, "l5": 96, "l6": 90, "l7": 85 },
    "evidence": [
      "High-confidence authorship match across research history.",
      "Repeated signal in low-latency ranking and retrieval systems.",
      "No proxy publishing or affiliation anomalies detected."
    ],
    "timeline": [
      { "time": "09:12", "label": "Profile ingested", "type": "info" },
      { "time": "09:18", "label": "L4 evaluation exceeded benchmark", "type": "success" },
      { "time": "09:24", "label": "Integrity verification complete", "type": "success" }
    ]
  }
]
```

> [!NOTE]
> The response should be a **flat array** (not paginated envelope). The frontend uses the array directly. If you need to add pagination metadata, wrap it — but you'll need to update the hooks to read `response.items` or similar.

---

### 8.2 `GET /candidates/:id`

**Used by:** `useCandidate(id)` hook → `EvidenceDrawer`, `CandidateDetail` page  
**Trigger:** When `activeCandidateId` is set (React Query key: `['candidate', id]`)  
**Enabled guard:** `enabled: Boolean(candidateId)` — only fires when an ID is present.

#### Example Request
```
GET /candidates/cand-alexei-kozlov
Authorization: Bearer <jwt>
```

#### Expected Response — `200 OK`
Same single-object schema as one item from the `/candidates` array (see §8.1).

#### Error — `404 Not Found`
```json
{ "detail": "Candidate not found." }
```

---

### 8.3 `POST /candidates/:id/schedule-interview`

**Used by:** `scheduleInterview(candidateId)` → `EvidenceDrawer` (Schedule Interview button)  
**Trigger:** User clicks "Schedule Interview" in the Evidence Drawer.

#### Example Request
```
POST /candidates/cand-alexei-kozlov/schedule-interview
Authorization: Bearer <jwt>
Content-Type: application/json
```
> No request body required; the action is implicit from the candidate ID.

#### Expected Response — `200 OK`
```json
{
  "ok": true,
  "candidate_id": "cand-alexei-kozlov",
  "message": "Alexei Kozlov moved to interview queue."
}
```

**Side effect in frontend:** After the response, the app navigates to `activePage = 'detail'` (Candidate Dossier page).

---

### 8.4 `GET /pipeline/runs`

**Used by:** `usePipelineRuns()` hook → `Overview` page  
**Trigger:** On Overview mount (React Query key: `['pipeline-runs']`)

#### Example Request
```
GET /pipeline/runs
Authorization: Bearer <jwt>
```

#### Expected Response — `200 OK`
```json
[
  {
    "id": "run-2026-0626",
    "job_title": "Senior AI Research Scientist",
    "status": "complete",
    "total_processed": 12847,
    "l1_rejects": 8421,
    "survivors": 4426,
    "shortlist_count": 214,
    "created_at": "2026-06-26T04:30:00Z"
  }
]
```

**Frontend usage:** `runs[0]` is used (most recent run). Fields `total_processed`, `l1_rejects`, `survivors`, `shortlist_count` populate the KPI cards. The run's `id` is passed to `/pipeline/runs/:id/funnel`.

---

### 8.5 `GET /pipeline/runs/:runId/funnel`

**Used by:** `usePipelineFunnel(runId)` hook → `Overview` page (`FunnelChart`)  
**Trigger:** After pipeline runs are fetched and `run.id` is available (React Query key: `['pipeline-funnel', runId]`)  
**Enabled guard:** `enabled: Boolean(runId)`

#### Example Request
```
GET /pipeline/runs/run-2026-0626/funnel
Authorization: Bearer <jwt>
```

#### Expected Response — `200 OK`
```json
[
  { "layer": "L1", "label": "Corpus Gate",   "count_in": 12847, "count_out": 4426 },
  { "layer": "L2", "label": "Signal Fit",    "count_in": 4426,  "count_out": 2104 },
  { "layer": "L3", "label": "Depth Rank",    "count_in": 2104,  "count_out": 1042 },
  { "layer": "L4", "label": "Integrity",     "count_in": 1042,  "count_out": 611  },
  { "layer": "L5", "label": "Trajectory",    "count_in": 611,   "count_out": 388  },
  { "layer": "L6", "label": "Domain Match",  "count_in": 388,   "count_out": 214  },
  { "layer": "L7", "label": "Reviewer Queue","count_in": 214,   "count_out": 214  }
]
```

**Frontend usage:** Each object renders as one horizontal bar in `FunnelChart.jsx`. The bar width is `(count_out / count_in) * 100%`.

---

### 8.6 `GET /system/health`

**Used by:** `useSystemHealth()` hook → `Overview` page, `SystemHealth` page  
**Trigger:** On mount (React Query key: `['system-health']`)

#### Example Request
```
GET /system/health
Authorization: Bearer <jwt>
```

#### Expected Response — `200 OK`
```json
{
  "neural_load": 72,
  "cache_size": "1.8 TB",
  "active_nodes": "142 / 142",
  "uptime": "99.982%",
  "latency_ms": 42,
  "queue_depth": 18,
  "core_efficiency": 98
}
```

**Frontend usage:**
- `latency_ms` → status tile in Overview header
- `neural_load`, `core_efficiency` → progress bars in SystemHealth
- `uptime`, `active_nodes` → status strips and topology card

---

### 8.7 `GET /system/events`

**Used by:** `useSystemEvents()` hook → `SystemHealth` page (Execution Log)  
**Trigger:** On mount (React Query key: `['system-events']`)

#### Query Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `limit` | `number` | `10` | Maximum number of events to return |

#### Example Request
```
GET /system/events?limit=10
Authorization: Bearer <jwt>
```

#### Expected Response — `200 OK`
```json
[
  {
    "id": "evt-1",
    "type": "patch",
    "message": "Ranking model patch v2.4.1 deployed",
    "author": "Admin_04",
    "created_at": "14:22 UTC"
  },
  {
    "id": "evt-2",
    "type": "warning",
    "message": "DB connection spike auto-corrected",
    "author": "Auto-fix",
    "created_at": "11:05 UTC"
  }
]
```

**Event types:** `"info"` | `"warning"` | `"patch"`  
**Icon mapping:**  
- `warning` → `AlertTriangle`  
- `patch` → `RotateCcw`  
- `info` (default) → `Terminal`

---

## 9. Data Models

### Candidate Object (full)

```ts
interface Candidate {
  id: string                     // Unique slug, e.g. "cand-alexei-kozlov"
  rank: string                   // Display rank, e.g. "01", "02"
  initials: string               // 2-letter avatar initials, e.g. "AK"
  name: string                   // Full name
  role: string                   // Job title
  background: string             // Education/experience summary
  location: string               // City, country
  domain: string                 // "core-ml" | "systems" | "alignment"
  score: number                  // Composite score 0–100
  match: string                  // Match percentage display, e.g. "98.4%"
  integrity_status: string       // "CLEAN" | "L_PROXY" | "FLAGGED"
  dominant_trait: string         // Primary identified strength
  verified_at: string            // ISO 8601 timestamp

  logic_scores: {
    l2: number                   // Knowledge layer score
    l3: number                   // Reasoning layer score
    l4: number                   // Technical layer score
    l5: number                   // Systems layer score
    l6: number                   // Domain layer score
    l7: number                   // Reviewer queue score
  }

  evidence: string[]             // 3 short evidence strings

  timeline: Array<{
    time: string                 // Display time, e.g. "09:12"
    label: string                // Event description
    type: "info" | "success" | "warning"
  }>
}
```

### PipelineRun Object

```ts
interface PipelineRun {
  id: string                     // e.g. "run-2026-0626"
  job_title: string
  status: "running" | "complete" | "failed"
  total_processed: number
  l1_rejects: number
  survivors: number
  shortlist_count: number
  created_at: string             // ISO 8601
}
```

### PipelineFunnelRow Object

```ts
interface FunnelRow {
  layer: string                  // "L1" – "L7"
  label: string                  // Human-readable stage name
  count_in: number               // Candidates entering this stage
  count_out: number              // Candidates passing this stage
}
```

### SystemHealth Object

```ts
interface SystemHealth {
  neural_load: number            // Percentage 0–100
  cache_size: string             // Display string, e.g. "1.8 TB"
  active_nodes: string           // Display string, e.g. "142 / 142"
  uptime: string                 // Display string, e.g. "99.982%"
  latency_ms: number             // Integer milliseconds
  queue_depth: number            // Current queue length
  core_efficiency: number        // Percentage 0–100
}
```

### SystemEvent Object

```ts
interface SystemEvent {
  id: string
  type: "info" | "warning" | "patch"
  message: string
  author: string
  created_at: string             // Display string, e.g. "14:22 UTC"
}
```

---

## 10. Pages & Routes

The app uses **manual state-based routing** via Zustand (`activePage`), not React Router. There are no URL-based routes.

| `activePage` value | Component | Description |
|---|---|---|
| `'overview'` | `Overview.jsx` | Dashboard with KPIs, funnel, risk cards, shortlist preview |
| `'candidates'` | `RankedCandidates.jsx` | Ranked list + FilterRail + Evidence Drawer |
| `'detail'` | `CandidateDetail.jsx` | Full dossier for `activeCandidateId` |
| `'health'` | `SystemHealth.jsx` | System diagnostics and execution event log |

### Navigation Triggers

| User Action | Result |
|---|---|
| Sidebar: "Overview" | `setActivePage('overview')` |
| Sidebar: "Ranked Candidates" | `setActivePage('candidates')` |
| Sidebar: "Settings" | `setActivePage('health')` |
| Candidate row click | `setActiveCandidate(id)` → opens drawer |
| Drawer: "Dossier" | `setActivePage('detail')` |
| Drawer: "Schedule Interview" | API call → `setActivePage('detail')` |
| Dossier: "Back to candidates" | `setActivePage('candidates')` |
| Overview: "View Full Report" | `setActivePage('candidates')` |
| Overview: candidate row click | `setActiveCandidate(id)` + `setActivePage('candidates')` |

---

## 11. Component Tree

```
App
├── LoginPage                   (shown when authToken is null)
└── AppShell                    (shown when authenticated)
    ├── Sidebar
    ├── TopBar
    └── main > PageWrapper
        ├── Overview             (activePage === 'overview')
        │   ├── ShapeLandingHero
        │   ├── FunnelChart
        │   ├── RiskCards
        │   └── [shortlist table]
        │
        ├── RankedCandidates     (activePage === 'candidates')
        │   ├── FilterRail
        │   ├── CandidateTable
        │   └── EvidenceDrawer   (renders when drawerOpen && activeCandidateId)
        │
        ├── CandidateDetail      (activePage === 'detail')
        │   ├── ScrollShowcase
        │   ├── ProgressBar (×many)
        │   └── Badge (×2)
        │
        └── SystemHealth         (activePage === 'health')
            ├── ScrollShowcase
            ├── ProgressBar (×3)
            └── [event list]
```

---

## 12. Filters & Query Parameters

The `FilterRail` component controls the global filter state. Every filter change triggers a new API call to `GET /candidates` via React Query (the query key includes the entire filters object).

| Filter | UI Control | Store key | API param |
|---|---|---|---|
| Score threshold | Range slider (60–98) | `filters.score_min` | `score_min` |
| Verified only | Checkbox | `filters.verified_only` | `verified_only` |
| Domain | Select + chip buttons | `filters.domain` | `domain` |
| Page | (internal) | `filters.page` | `page` |
| Limit | (internal) | `filters.limit` | `limit` |

**Domain chip mappings:**

| Chip Label | Domain Value |
|---|---|
| LLM Architecture | `core-ml` |
| NLP Eng | `systems` |
| Cloud Infra | `systems` |
| PyTorch | `alignment` |

---

## 13. Production Backend Integration Guide

### Step 1: Set Environment Variables

```env
VITE_API_URL=https://api.yourbackend.com    # Your production API base URL
VITE_USE_MOCK_API=false                      # Switch off mock data
VITE_SUPABASE_URL=https://xyz.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key
```

### Step 2: Implement Required Endpoints

Your backend must expose the following routes. All endpoints must:
- Accept `Authorization: Bearer <jwt>` header
- Validate the JWT (Supabase JWT or your own)
- Return `401` for invalid/expired tokens (the frontend will auto-logout)
- Return `Content-Type: application/json`

| Method | Path | Required |
|---|---|---|
| `GET` | `/candidates` | ✅ Yes |
| `GET` | `/candidates/:id` | ✅ Yes |
| `POST` | `/candidates/:id/schedule-interview` | ✅ Yes |
| `GET` | `/pipeline/runs` | ✅ Yes |
| `GET` | `/pipeline/runs/:id/funnel` | ✅ Yes |
| `GET` | `/system/health` | ✅ Yes |
| `GET` | `/system/events` | ✅ Yes |

### Step 3: CORS Configuration

Your backend must allow requests from your frontend's origin:

```
Access-Control-Allow-Origin: https://your-frontend-domain.com
Access-Control-Allow-Headers: Authorization, Content-Type
Access-Control-Allow-Methods: GET, POST, OPTIONS
```

For local development, also allow: `http://localhost:5173`

### Step 4: JWT Validation (Supabase)

If using Supabase Auth:
1. Get your Supabase project's **JWT secret** from the Supabase dashboard
2. Validate incoming JWTs using that secret
3. The `sub` field in the JWT payload is the user's UUID

Example (Python/FastAPI):
```python
from jose import JWTError, jwt

SUPABASE_JWT_SECRET = os.environ["SUPABASE_JWT_SECRET"]

def verify_token(token: str):
    payload = jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"])
    return payload
```

### Step 5: Candidates Endpoint Requirements

The `/candidates` endpoint must support filtering:

```python
# Example Python pseudo-code
@app.get("/candidates")
def list_candidates(
    score_min: int = 70,
    verified_only: bool = True,
    domain: str = "all",
    page: int = 1,
    limit: int = 20,
    token: str = Depends(verify_token)
):
    query = db.candidates
    query = query.filter(score >= score_min)
    if verified_only:
        query = query.filter(integrity_status == "CLEAN")
    if domain != "all":
        query = query.filter(domain == domain)
    query = query.order_by(score.desc())
    query = query.offset((page - 1) * limit).limit(limit)
    return query.all()
```

### Step 6: Schedule Interview Webhook

The `POST /candidates/:id/schedule-interview` endpoint should:
1. Create an interview record in your database
2. Optionally trigger a notification (email, calendar invite, Slack)
3. Return the standard response format (see §8.3)

### Step 7: System Health Endpoint

The `/system/health` endpoint should return live metrics from your infrastructure. Field mapping:

| Field | Source |
|---|---|
| `neural_load` | ML worker CPU/GPU utilization % |
| `cache_size` | Redis/vector DB cache size |
| `active_nodes` | Worker node count (running / total) |
| `uptime` | Service uptime % (past 30 days) |
| `latency_ms` | Average API response time in ms |
| `queue_depth` | Length of candidate processing queue |
| `core_efficiency` | Custom efficiency score 0–100 |

### Step 8: Verify Integration

After deploying your backend:

1. Set `VITE_USE_MOCK_API=false` in your `.env`
2. Run `npm run dev`
3. Log in — the JWT should be stored and sent with every request
4. Open DevTools Network tab, verify:
   - All requests go to your `VITE_API_URL`
   - `Authorization: Bearer ...` header is present
   - Responses match the schemas in §8
5. Test the 401 flow: invalidate the token and confirm auto-logout fires

---

## 14. Build & Deployment

### Development

```bash
npm install
npm run dev
# Starts on http://127.0.0.1:5173
```

### Production Build

```bash
npm run build
# Outputs to /dist
```

The Vite config splits the bundle into 4 named chunks for optimal loading:

| Chunk | Contents |
|---|---|
| `react` | `react`, `react-dom` |
| `three` | `three`, `@react-three/fiber`, `@react-three/drei`, `@react-spring/three` |
| `motion` | `framer-motion` |
| `data` | `@tanstack/react-query`, `zustand`, `axios`, `@supabase/supabase-js` |

### Serving the SPA

Since this is a SPA with no URL-based routing, configure your server to serve `index.html` for all routes:

**Nginx:**
```nginx
location / {
  try_files $uri $uri/ /index.html;
}
```

**Apache:**
```apache
FallbackResource /index.html
```

**Vercel / Netlify:** Add a `_redirects` file (Netlify) or `vercel.json`:
```json
{ "rewrites": [{ "source": "/(.*)", "destination": "/index.html" }] }
```

### Preview Built App

```bash
npm run preview
# Serves /dist on http://127.0.0.1:4173
```

---

## 15. Design System Notes

### CSS Architecture

All styles are in a single file: `src/styles.css`  
Uses **CSS custom properties** (variables) for the full design token system.

### Key Design Tokens

```css
/* Primary accent — electric teal */
--accent:     #2dd4aa;
--accent-dim: #1a8c72;

/* Backgrounds */
--bg:         #0a0f0d;    /* Page background */
--surface:    #111a16;    /* Card/panel background */
--surface-2:  #172219;    /* Elevated surface */
--border:     #1f2e28;    /* Subtle borders */

/* Text */
--text:       #e8f5f0;    /* Primary text */
--text-2:     #8aaa9f;    /* Secondary/muted text */

/* Status colors */
--success:    #2dd4aa;
--warning:    #f59e0b;
--danger:     #ef4444;
```

### Typography

- **Font:** System stack (no Google Fonts import in current version)
- **Heading tracking:** `letter-spacing: 0.05em` on section labels
- **ALL CAPS labels:** Used extensively for eyebrow text and nav items

### Animation Libraries

| Usage | Library |
|---|---|
| Page transitions (fade/slide) | `framer-motion` (AnimatePresence + motion.div) |
| Evidence Drawer slide-in | `framer-motion` (x: 420 → 0) |
| 3D scenes (hero, score orb) | `@react-three/fiber` + `@react-spring/three` |
| Login particle background | `@tsparticles/react` + `@tsparticles/slim` |
| Progress bar fill animation | CSS transitions |

### Dark Mode

- Toggled by adding/removing `.dark` class on `<html>`
- Persisted in `localStorage` as `redrob_dark_mode`
- The current theme is always dark-first; light mode tokens are defined in `:root.dark` override block

---

*End of RedRob Frontend Technical Reference*
