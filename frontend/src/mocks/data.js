export const candidates = [
  {
    id: 'cand-alexei-kozlov',
    rank: '01',
    initials: 'AK',
    name: 'Alexei Kozlov',
    role: 'Principal ML Systems Researcher',
    background: 'PhD, Stanford - 8 years experience',
    location: 'San Francisco, CA',
    domain: 'core-ml',
    score: 94,
    match: '98.4%',
    integrity_status: 'CLEAN',
    dominant_trait: 'Core Algorithm',
    verified_at: '2026-06-26T04:40:00Z',
    logic_scores: { l2: 94, l3: 88, l4: 92, l5: 96, l6: 90, l7: 85 },
    required_skills: ['PyTorch', 'Transformers', 'Python', 'Distributed Training', 'LLM Research'],
    inferred_skills: ['CUDA Optimization', 'Low-latency Systems', 'Retrieval Architectures', 'Edge Deployment'],
    evidence: [
      'High-confidence authorship match across research history.',
      'Repeated signal in low-latency ranking and retrieval systems.',
      'No proxy publishing or affiliation anomalies detected.'
    ],
    timeline: [
      { time: '09:12', label: 'Profile ingested', type: 'info' },
      { time: '09:18', label: 'L4 evaluation exceeded benchmark', type: 'success' },
      { time: '09:24', label: 'Integrity verification complete', type: 'success' }
    ]
  },
  {
    id: 'cand-sarah-chen',
    rank: '02',
    initials: 'SC',
    name: 'Sarah Chen',
    role: 'Distributed AI Infrastructure Lead',
    background: 'Ex-Google - 6 years experience',
    location: 'Seattle, WA',
    domain: 'systems',
    score: 91,
    match: '97.1%',
    integrity_status: 'CLEAN',
    dominant_trait: 'Systems Engineering',
    verified_at: '2026-06-26T03:50:00Z',
    logic_scores: { l2: 88, l3: 95, l4: 82, l5: 88, l6: 92, l7: 90 },
    required_skills: ['Python', 'Distributed Training', 'LLM Research', 'Kubernetes'],
    inferred_skills: ['ML Ops', 'Observability', 'Platform Reliability', 'Cloud Infra'],
    evidence: [
      'Strong distributed training and observability record.',
      'Consistent release velocity across infra-heavy programs.',
      'Domain match is strongest for platform reliability roles.'
    ],
    timeline: [
      { time: '09:04', label: 'Profile ingested', type: 'info' },
      { time: '09:15', label: 'Systems evidence linked', type: 'success' },
      { time: '09:22', label: 'Shortlist threshold cleared', type: 'success' }
    ]
  },
  {
    id: 'cand-jordan-taylor',
    rank: '03',
    initials: 'JT',
    name: 'Jordan Taylor',
    role: 'Applied Alignment Scientist',
    background: 'OpenAI Labs - 4 years experience',
    location: 'New York, NY',
    domain: 'alignment',
    score: 87,
    match: '96.8%',
    integrity_status: 'L_PROXY',
    dominant_trait: 'Publication Velocity',
    verified_at: '2026-06-25T18:30:00Z',
    logic_scores: { l2: 82, l3: 80, l4: 85, l5: 91, l6: 88, l7: 84 },
    required_skills: ['Python', 'LLM Research', 'Transformers'],
    inferred_skills: ['RLHF', 'Evaluation Design', 'Safety Research', 'Red-teaming'],
    evidence: [
      'Excellent evaluation design signal.',
      'One low-confidence co-author network requires human review.',
      'Best fit for applied safety research, not infra ownership.'
    ],
    timeline: [
      { time: '08:44', label: 'Profile ingested', type: 'info' },
      { time: '08:52', label: 'Proxy signal flagged', type: 'warning' },
      { time: '09:06', label: 'Reviewer note attached', type: 'warning' }
    ]
  },
  {
    id: 'cand-priya-menon',
    rank: '04',
    initials: 'PM',
    name: 'Priya Menon',
    role: 'Research Engineering Manager',
    background: 'DeepMind - 5 years experience',
    location: 'London, UK',
    domain: 'core-ml',
    score: 85,
    match: '94.2%',
    integrity_status: 'CLEAN',
    dominant_trait: 'ML Research',
    verified_at: '2026-06-25T15:18:00Z',
    logic_scores: { l2: 85, l3: 82, l4: 78, l5: 88, l6: 82, l7: 80 },
    required_skills: ['PyTorch', 'Transformers', 'Python', 'LLM Research'],
    inferred_skills: ['Research Management', 'Publication Strategy', 'Cross-functional Delivery'],
    evidence: [
      'Strong publication quality with durable citation signal.',
      'Excellent cross-functional delivery history.',
      'Slightly lower systems score than top two candidates.'
    ],
    timeline: [
      { time: '08:18', label: 'Profile ingested', type: 'info' },
      { time: '08:26', label: 'Leadership evidence ranked', type: 'success' },
      { time: '08:40', label: 'Shortlist threshold cleared', type: 'success' }
    ]
  },
  {
    id: 'cand-leon-weber',
    rank: '05',
    initials: 'LW',
    name: 'Leon Weber',
    role: 'Alignment Evaluation Researcher',
    background: 'Anthropic - 3 years experience',
    location: 'Berlin, DE',
    domain: 'alignment',
    score: 83,
    match: '93.9%',
    integrity_status: 'CLEAN',
    dominant_trait: 'Alignment',
    verified_at: '2026-06-25T11:42:00Z',
    logic_scores: { l2: 80, l3: 84, l4: 78, l5: 86, l6: 82, l7: 79 },
    required_skills: ['Python', 'LLM Research', 'Transformers'],
    inferred_skills: ['Alignment Research', 'Evaluation Design', 'Red-teaming', 'Safety Probing'],
    evidence: [
      'Best evidence appears in eval design and red-team loops.',
      'Career depth is lower but trajectory is strong.',
      'Clean identity and affiliation verification.'
    ],
    timeline: [
      { time: '07:55', label: 'Profile ingested', type: 'info' },
      { time: '08:11', label: 'Evaluation corpus matched', type: 'success' },
      { time: '08:19', label: 'Identity verification complete', type: 'success' }
    ]
  }
]

export const pipelineRuns = [
  {
    id: 'run-2026-0626',
    job_title: 'Senior AI Research Scientist',
    status: 'complete',
    total_processed: 12847,
    l1_rejects: 8421,
    survivors: 4426,
    shortlist_count: 214,
    created_at: '2026-06-26T04:30:00Z'
  }
]

export const pipelineFunnel = [
  { layer: 'L1', label: 'Corpus Gate', count_in: 12847, count_out: 4426 },
  { layer: 'L2', label: 'Signal Fit', count_in: 4426, count_out: 2104 },
  { layer: 'L3', label: 'Depth Rank', count_in: 2104, count_out: 1042 },
  { layer: 'L4', label: 'Integrity', count_in: 1042, count_out: 611 },
  { layer: 'L5', label: 'Trajectory', count_in: 611, count_out: 388 },
  { layer: 'L6', label: 'Domain Match', count_in: 388, count_out: 214 },
  { layer: 'L7', label: 'Reviewer Queue', count_in: 214, count_out: 214 }
]

export const systemHealth = {
  neural_load: 72,
  cache_size: '1.8 TB',
  active_nodes: '142 / 142',
  uptime: '99.982%',
  latency_ms: 42,
  queue_depth: 18,
  core_efficiency: 98
}

export const systemEvents = [
  { id: 'evt-1', type: 'patch', message: 'Ranking model patch v2.4.1 deployed', author: 'Admin_04', created_at: '14:22 UTC' },
  { id: 'evt-2', type: 'warning', message: 'DB connection spike auto-corrected', author: 'Auto-fix', created_at: '11:05 UTC' },
  { id: 'evt-3', type: 'info', message: 'Supabase token refresh validated', author: 'Auth monitor', created_at: '10:18 UTC' },
  { id: 'evt-4', type: 'info', message: 'Candidate snapshot verified', author: 'Scheduler', created_at: '08:00 UTC' }
]
