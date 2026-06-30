# RedRob Frontend Context And Flow

## Current Objective

The RedRob frontend has been changed from a login-first command center into a complete shortlisting flow:

1. Input upload page opens first.
2. User provides a candidate archive and job description.
3. Processing page opens with a live audit sidebar and interactive game zone.
4. When processing completes, the new command-center overview opens.
5. Overview uses a horizontal animated L1-L9 layer card stack.

This document describes what has been built, how it works, and how to test or export it.

## Visual References Used

The new input, processing, and overview direction was based on screenshots extracted from these zip files:

- `C:\Redrobai\stitch_fuzzy_ai_candidate_shortlister_input.zip`
- `C:\Redrobai\stitch_fuzzy_ai_candidate_shortlister_loading.zip`
- `C:\Redrobai\stitch_redrob_ai_recruitment_interface_overview.zip`
- `C:\Redrobai\stitch_screenshot_page_assembler_lightsystem.zip`
- `C:\Redrobai\stitch_screenshot_page_assemblersystem.zip`

Only image/screenshot files were inspected from these Stitch reference zips. Source code from those zips was not used.

## App Structure

Main app flow is controlled from:

- `src/App.jsx`
- `src/store/useAppStore.js`

The Zustand store now includes:

- `flowStage`: `input`, `processing`, or `app`
- `uploadedFiles`: candidate archive and job description metadata
- `processingProgress`: numeric processing progress
- `activePage`: command-center page after processing
- `darkMode`: global theme
- `filters`, `activeCandidateId`, and `drawerOpen`: existing recruitment UI state

## Flow Details

### 1. Input Page

File:

- `src/pages/InputPage.jsx`

Purpose:

- Replaces the old login page.
- Opens first on page load.
- Accepts candidate archive and job description files.

UI:

- Dark starfield background.
- Centered console panel.
- `Welcome` heading.
- `Fuzzy AI Reasoning Shortlister` subtitle.
- Candidate archive upload box.
- Job description upload box.
- Large `Start Shortlisting` button.

Accepted files:

- Candidate archive: `.zip`
- Job description: `.pdf`, `.doc`, `.docx`

Behavior:

- User can click upload boxes or drag/drop files.
- Start button remains disabled until both files are selected.
- On start, the store calls `startProcessing(files)` and moves to the processing stage.

Animation:

- Uses `Sparkles` particle field from `src/components/ui/Sparkles.jsx`.

### 2. Processing Page

File:

- `src/pages/ProcessingPage.jsx`

Purpose:

- Simulates the AI shortlisting process after file upload.
- Shows the user that analysis is happening.

UI:

- Framed Fuzzy AI processing shell.
- Header with system load progress.
- Left sidebar audit log.
- Large black game zone.
- Footer with session ID.

Audit log:

- Initializing neural pathways
- Loading candidate database
- Parsing job description vectors
- Normalizing career timeline artifacts
- Running L1-L9 survival heuristics
- Resolving duplicate identity clusters
- Scoring collaboration entropy
- Synthesizing final shortlist

Game zone:

- Named `Neural Runner`.
- User can click or press `Space` to jump.
- Score increases while active.
- The processing progress advances automatically.

Completion:

- When progress reaches 100, `completeProcessing()` moves the app into the command center.
- The active command-center page is `overview`.
- Dark mode is enabled to match the new overview screenshots.

### 3. Command Center Shell

Files:

- `src/components/layout/AppShell.jsx`
- `src/components/layout/Sidebar.jsx`
- `src/components/layout/TopBar.jsx`

Purpose:

- Persistent shell after processing completes.

Sidebar:

- RedRob branding.
- Overview.
- Job Intake.
- Pipeline Run.
- Ranked Candidates.
- Analytics.
- Settings.
- New Pipeline button.

Top bar:

- Page title.
- Search field.
- Notifications icon.
- Theme toggle.
- Admin/Preview pill.
- Sign out icon.

Sign out behavior:

- Resets app back to the input/upload page.

### 4. New Overview Page

File:

- `src/pages/Overview.jsx`

Purpose:

- Shows the post-processing result dashboard.
- Imitates the new overview reference screenshot.

Key elements:

- `Overview` heading.
- System status and latency tile.
- KPI cards:
  - Total processed
  - L1 rejects
  - Gate survivors
  - Final shortlist
- Pipeline Layer Stack.
- Right metric rail:
  - Total candidates
  - Hard rejects
  - Processed
  - Output
- Best candidate feature panel.
- Risk and sparsity panels.
- Top performers preview.

### 5. Horizontal L1-L9 Card Stack

File:

- `src/components/overview/HorizontalLayerStack.jsx`

Purpose:

- Replaces the vertical card stack reference with the requested horizontal card stack.
- Represents layers L1-L9.

Layers:

- L1 Input Layer
- L2 Identity Gate
- L3 Resume Parse
- L4 Skill Match
- L5 Experience Fit
- L6 Integrity Scan
- L7 Reasoning Core
- L8 Human Review
- L9 Output

Behavior:

- Cards are visually stacked horizontally.
- Top/front card can be dragged horizontally.
- Clicking a card rotates it through the stack.
- Framer Motion spring animation controls position, scale, brightness, and drag feedback.

Animation source:

- Adapted from the provided stack-card concept.
- Changed from vertical stack behavior to horizontal stack behavior.
- Uses app-native data cards instead of external images.

### 6. Existing Pages Still Present

The following pages still exist after the overview:

- Ranked Candidates
- Candidate Detail
- System Health

They still use:

- Zustand state.
- TanStack Query mock/live API pattern.
- Evidence drawer.
- Filters.
- Theme toggle.
- Page transitions.

## Important Files Changed Or Added

Added:

- `src/pages/InputPage.jsx`
- `src/pages/ProcessingPage.jsx`
- `src/components/overview/HorizontalLayerStack.jsx`
- `docs/RedRob_Frontend_Context_And_Flow.md`

Changed:

- `src/App.jsx`
- `src/store/useAppStore.js`
- `src/pages/Overview.jsx`
- `src/styles.css`
- `.gitignore`

Previously added and still used:

- `src/components/ui/Sparkles.jsx`
- `src/components/ui/ScrollShowcase.jsx`
- `src/components/ui/ShapeLandingHero.jsx`
- `docs/RedRob_Frontend_Planner.md`

## Local Testing Steps In VS Code

1. Open VS Code.
2. Open folder:

```text
C:\Users\KIIT0001\Documents\redrob
```

3. Open terminal:

```text
Terminal > New Terminal
```

4. Install dependencies:

```powershell
npm install
```

5. Start dev server:

```powershell
npm run dev
```

6. Open:

```text
http://127.0.0.1:5173/
```

7. Confirm input page appears first.

8. Upload/select:

- Any `.zip` for candidate archive.
- Any `.pdf`, `.doc`, or `.docx` for job description.

9. Click `Start Shortlisting`.

10. On processing page:

- Confirm audit log appears.
- Click the game zone or press `Space`.
- Confirm progress moves.
- Wait for completion.

11. On overview:

- Confirm dark command-center layout.
- Confirm horizontal L1-L9 card stack.
- Drag/click cards to rotate the stack.
- Confirm metric rail and candidate panel appear.

12. Test navigation:

- Ranked Candidates.
- Candidate Detail.
- Settings/System Health.

13. Test theme:

- Use top-right theme toggle.

14. Test build:

```powershell
npm run build
```

Expected result:

- Vite build completes successfully.

## GitHub Export From VS Code

1. Open Source Control panel in VS Code.
2. Review changed files.
3. Do not commit:

- `.env`
- `node_modules`
- `dist`
- `artifacts`
- `reference-images`
- `reference-flow-images`

These are already ignored in `.gitignore`.

4. Stage desired files.
5. Commit:

```text
Add RedRob upload-processing-overview flow
```

6. Click `Publish Branch`.
7. Sign into GitHub if prompted.
8. Choose private/public repository.
9. Verify GitHub contains:

- Source files.
- `package.json`
- `.env.example`
- `docs/RedRob_Frontend_Context_And_Flow.md`
- `docs/RedRob_Frontend_Planner.md`

10. Verify GitHub does not contain:

- `.env`
- local screenshots/reference extraction folders
- `node_modules`
- `dist`

## Future Work Plan

Recommended next improvements:

1. Replace simulated processing with a real upload endpoint.
2. Store uploaded candidate archive and job description in backend storage.
3. Stream real audit log events from FastAPI using SSE or WebSockets.
4. Replace the game-zone placeholder with a richer branded runner or puzzle.
5. Add upload validation for file size and file type.
6. Add error state if processing fails.
7. Add a processing results summary before entering Overview.
8. Connect L1-L9 layer cards to real pipeline layer metrics.
9. Add keyboard accessibility labels for the horizontal card stack.
10. Add Playwright tests for:
    - Upload flow
    - Processing completion
    - Horizontal layer stack interaction
    - Overview render
    - Theme toggle
    - Candidate drawer

## Verification Status

Latest build command:

```powershell
npm run build
```

Status:

- Passed after the new flow implementation.
