# RedRob Frontend Planner

## Purpose

RedRob is a command-center style AI recruitment interface. The frontend presents candidate ingestion, ranking, evidence review, dossier inspection, and system diagnostics in one authenticated React/Vite application.

The visual direction now follows the supplied PNG reference screens from `C:\Redrobai\stitch_redrob_ai_recruitment_interface.zip`. Only PNG/image assets were inspected from that zip; source code inside the zip was intentionally not used.

## Visual Direction

The interface should feel like a precise operational console rather than a marketing dashboard.

Core traits:

- Fixed left navigation with RedRob branding, mono uppercase labels, teal active states, and a bottom pipeline action.
- Thin top command bar with page title, global search, notifications, theme toggle, admin status, and sign out.
- Off-white light mode and near-black dark mode.
- Low-radius panels with thin borders and restrained shadows.
- Dense but readable layouts for repeated operational use.
- Teal is the primary action/status color, with red for risk and amber for sync/warning states.
- Serif italic copy is used only for intelligence notes, model summaries, and operational commentary.

## Major Pages

### Login

Purpose: Auth gate before the command center.

Behavior:

- Uses Supabase sign-in when real `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY` are configured.
- Uses local preview login when Supabase values are placeholders.
- Stores the access token in `localStorage` as `supabase_token`.

Animation:

- Sparkles particle field across the login page.
- Horizontal light beam behind the login panel.

### Overview

Purpose: System-wide pipeline overview.

Key UI:

- Animated geometric landing hero with `Systems Overview` headline.
- Status tile for system health and latency.
- KPI strip: total processed, L1 rejects, survivors, final shortlist.
- Funnel visualization with layered L1/L2/L3/final shortlist blocks.
- Neural insights rail.
- Intake CTA.
- Risk/data sparsity panels.
- Top performers preview table.

Animation:

- Landing hero shapes enter and float.
- Funnel panel is wrapped in a scroll-tilt showcase.
- Page transition uses Framer Motion.

### Ranked Candidates

Purpose: Filter and review the ranked candidate shortlist.

Key UI:

- Left filter rail with score range, verified-only toggle, risk option, domain chips, and reset.
- Center ranked candidate stack.
- Right evidence drawer with logic breakdown cards, model confidence, synthesis fit, and evidence notes.

Behavior:

- Filter controls update Zustand filter state.
- TanStack Query refetches candidate data using the filter key.
- Candidate row click updates `activeCandidateId` and opens the evidence drawer.
- Schedule Interview moves to the dossier page.

Animation:

- Evidence drawer slides in with Framer Motion.
- Progress bars animate through CSS transitions.

### Candidate Detail

Purpose: Full candidate dossier.

Key UI:

- Candidate profile header with rank/verified badges, location, verification date, and skill tags.
- Vetting velocity score panel.
- Neural signature panel.
- Atmospheric signal cards.
- Verified professional evidence panel.
- Recommendation quote and action buttons.
- Timeline fallback panel for smaller/detail review.

Animation:

- Evidence panel uses the reusable scroll-tilt animation.
- Page transition uses Framer Motion.

### System Health

Purpose: Operational diagnostics/control surface.

Key UI:

- Global engine and latency threshold status strips.
- Architecture health integrity report.
- Diagnostic metrics for neural load, cache memory, and environment threads.
- Core configuration toggles.
- Knowledge base status.
- Core efficiency score.
- Execution log.
- Network topology indicator.

Animation:

- Integrity report is wrapped in the scroll-tilt showcase.
- Topology bars remain visually active and compact.

## State And Data Flow

Main files:

- `src/store/useAppStore.js`: Zustand store for active page, theme, auth token, active candidate, drawer state, and filters.
- `src/api/client.js`: Axios client with JWT injection and 401 handling.
- `src/api/redrobApi.js`: API facade that uses live FastAPI when configured and mock fallback otherwise.
- `src/mocks/data.js`: Local preview data.
- `src/hooks/useCandidates.js`: Candidate list/detail query hooks.
- `src/hooks/usePipeline.js`: Pipeline run/funnel query hooks.
- `src/hooks/useSystemHealth.js`: Health and event query hooks.

## Animation Components

- `src/components/ui/ShapeLandingHero.jsx`
  - Used on Overview.
  - Floating geometric shapes with staged headline entrance.

- `src/components/ui/Sparkles.jsx`
  - Used on Login.
  - Powered by `@tsparticles/react` and `@tsparticles/slim`.

- `src/components/ui/ScrollShowcase.jsx`
  - Used on Overview, Candidate Detail, and System Health.
  - Uses `useScroll`, `useTransform`, and `motion.div` to tilt/scale panels as the page scrolls.

- `src/components/layout/PageWrapper.jsx`
  - Wraps pages with fade/slide transitions.

## Environment Setup

Create `.env` from `.env.example`.

Local preview:

```env
VITE_API_URL=http://localhost:8000
VITE_USE_MOCK_API=true
VITE_SUPABASE_URL=https://xxxx.supabase.co
VITE_SUPABASE_ANON_KEY=replace-with-supabase-anon-key
```

Live backend/auth:

```env
VITE_API_URL=https://your-backend-url
VITE_USE_MOCK_API=false
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key
```

## Run Locally In VS Code

1. Open VS Code.
2. Select `File > Open Folder`.
3. Open `C:\Users\KIIT0001\Documents\redrob`.
4. Open `Terminal > New Terminal`.
5. Install dependencies:

```powershell
npm install
```

6. Create `.env`:

```powershell
copy .env.example .env
```

7. Start the dev server:

```powershell
npm run dev
```

8. Open the local URL shown in the terminal, usually:

```text
http://127.0.0.1:5173/
```

9. Use local preview credentials:

```text
admin@redrob.ai
preview
```

## Testing Checklist

### Build

```powershell
npm run build
```

Expected result: Vite build completes successfully.

### Login

- Login page renders.
- Sparkles animation is visible behind the login panel.
- Local preview credentials enter the app.

### Overview

- `Systems Overview` header renders.
- Landing hero shapes animate and float.
- KPI cards and funnel layers render.
- Scroll the page and confirm the funnel showcase tilts smoothly.
- Click `View Full Report` or a candidate row to open Ranked Candidates.

### Ranked Candidates

- Score range updates the list.
- Verified-only toggle updates the list.
- Domain chips update the list.
- Candidate click opens/updates the evidence drawer.
- `Schedule Interview` opens Candidate Detail.

### Candidate Detail

- Dossier card, vetting velocity, neural signature, evidence entries, and action buttons render.
- Scroll the page and confirm the evidence showcase tilts smoothly.
- `Back to candidates` returns to Ranked Candidates.

### System Health

- Status strips, integrity report, configuration panels, execution log, and topology render.
- Scroll the page and confirm the architecture health showcase tilts smoothly.

### Theme

- Toggle dark mode from the top bar.
- Confirm all pages remain readable and aligned.

### Responsive

- Test desktop width around `1280px`.
- Test mobile width around `390px`.
- Confirm sidebar becomes a horizontal nav, search wraps, panels stack, and the evidence drawer fits.

## Upload To GitHub From VS Code

1. Open the project folder in VS Code:

```text
C:\Users\KIIT0001\Documents\redrob
```

2. Open Source Control from the left sidebar.
3. Review changed files.
4. Do not commit `.env`.
5. Stage the files you want to upload.
6. Commit with a clear message:

```text
Match RedRob UI references and add animations
```

7. Click `Publish Branch`.
8. Sign in to GitHub if prompted.
9. Choose public/private repository visibility.
10. After publishing, verify the repository includes `.env.example` but not `.env`.

## Future Change Planner

Recommended next changes:

1. Replace mock data with FastAPI/Supabase data by setting `VITE_USE_MOCK_API=false`.
2. Add route-based navigation with React Router once URLs need to be shareable.
3. Add real user profile/avatar data from Supabase Auth.
4. Add persisted filter presets for recruiting teams.
5. Add accessible table mode for screen readers on the ranked candidates page.
6. Add server-driven pagination for candidate lists larger than 100 rows.
7. Add real audit/export endpoints for `Audit Trace` and `Export JSON`.
8. Add automated Playwright tests for login, filtering, drawer open, theme toggle, and responsive drawer behavior.
9. If deployment bundle size grows, lazy-load `@tsparticles` and the heavier visual animation surfaces.
10. Build a design-token file if the product expands beyond this single command center.
