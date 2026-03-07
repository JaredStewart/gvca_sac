const API_BASE = import.meta.env.VITE_API_URL || ''

interface RequestOptions {
  method?: string
  body?: unknown
  headers?: Record<string, string>
}

async function request<T>(endpoint: string, options: RequestOptions = {}): Promise<T> {
  const { method = 'GET', body, headers = {} } = options

  const config: RequestInit = {
    method,
    headers: {
      'Content-Type': 'application/json',
      ...headers,
    },
  }

  if (body) {
    config.body = JSON.stringify(body)
  }

  const response = await fetch(`${API_BASE}${endpoint}`, config)

  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(error.detail || `Request failed: ${response.status}`)
  }

  return response.json()
}

// Pipeline API
export const pipelineApi = {
  getYears: () => request<{ years: string[] }>('/api/pipeline/years'),

  init: (year: string, options?: { force_reprocess?: boolean; weight_by_parents?: boolean }) =>
    request<{ year: string; status: string; row_count: number }>(`/api/pipeline/init/${year}`, {
      method: 'POST',
      body: options || {},
    }),

  getStatus: (year: string) =>
    request<{
      year: string
      initialized: boolean
      loaded: boolean
      row_count: number
      tagging_complete: boolean
      embeddings_complete: boolean
      clustering_complete: boolean
      last_updated: string | null
    }>(`/api/pipeline/status/${year}`),

  getAllStatus: () =>
    request<{ years: YearStatus[] }>('/api/pipeline/status-all'),

  run: (year: string, options: {
    run_tagging?: boolean
    run_clustering?: boolean
    use_fragments?: boolean
    model?: string
    n_samples?: number
    threshold?: number
  }) =>
    request<{ job_id: string; year: string; status: string }>(`/api/pipeline/run/${year}`, {
      method: 'POST',
      body: options,
    }),
}

// Data API
export const dataApi = {
  // Import CSV file to database
  importData: async (year: string, file: File, replaceExisting = false): Promise<ImportResult> => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('replace_existing', String(replaceExisting))

    const response = await fetch(`${API_BASE}/api/data/import/${year}`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({}))
      throw new Error(error.detail || `Import failed: ${response.status}`)
    }

    return response.json()
  },

  // Export tagging results as CSV download
  exportTagging: async (year: string): Promise<void> => {
    const response = await fetch(`${API_BASE}/api/data/${year}/tagging-export`)
    if (!response.ok) {
      throw new Error(`Export failed: ${response.status}`)
    }
    const blob = await response.blob()
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `tagging_${year}.csv`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    window.URL.revokeObjectURL(url)
  },

  // Export data as CSV download
  exportData: async (year: string): Promise<void> => {
    const response = await fetch(`${API_BASE}/api/data/${year}/export`)
    if (!response.ok) {
      throw new Error(`Export failed: ${response.status}`)
    }
    const blob = await response.blob()
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `survey_data_${year}.csv`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    window.URL.revokeObjectURL(url)
  },

  // Get free response counts (Good Choice vs Better Serve)
  getFreeResponseCounts: (year: string) =>
    request<{
      total_good_choice: number
      total_better_serve: number
      only_good_choice: number
      only_better_serve: number
      both: number
      only_positive_pct: number
    }>(`/api/data/${year}/free-response-counts`),

  // Get survey responses from database with filters
  getDbResponses: (year: string, params?: {
    page?: number
    per_page?: number
    school_level?: string
    demographic?: string
    sort?: string
  }) => {
    const searchParams = new URLSearchParams()
    if (params?.page) searchParams.set('page', String(params.page))
    if (params?.per_page) searchParams.set('per_page', String(params.per_page))
    if (params?.school_level) searchParams.set('school_level', params.school_level)
    if (params?.demographic) searchParams.set('demographic', params.demographic)
    if (params?.sort) searchParams.set('sort', params.sort)

    return request<PaginatedResponse<SurveyResponse>>(
      `/api/data/${year}/db-responses?${searchParams}`
    )
  },

  // Get free-text responses from database
  getFreeResponses: (year: string, params?: {
    page?: number
    per_page?: number
    question_type?: 'praise' | 'improvement'
    level?: string
    question?: string
  }) => {
    const searchParams = new URLSearchParams()
    if (params?.page) searchParams.set('page', String(params.page))
    if (params?.per_page) searchParams.set('per_page', String(params.per_page))
    if (params?.question_type) searchParams.set('question_type', params.question_type)
    if (params?.level) searchParams.set('level', params.level)
    if (params?.question) searchParams.set('question', params.question)

    return request<PaginatedResponse<FreeResponse>>(
      `/api/data/${year}/free-responses?${searchParams}`
    )
  },

  load: (year: string, force_reprocess = false) =>
    request<{ year: string; status: string; row_count: number; columns: string[] }>(
      `/api/data/${year}/load`,
      { method: 'POST', body: { force_reprocess } }
    ),

  getStatistics: (year: string) =>
    request<{
      total_responses: number
      weighted_averages: Record<string, number | null>
      level_counts: Record<string, number>
      question_totals: unknown[]
    }>(`/api/data/${year}/statistics`),

  getResponses: (year: string, params?: {
    page?: number
    per_page?: number
    level?: string
    has_response?: boolean
  }) => {
    const searchParams = new URLSearchParams()
    if (params?.page) searchParams.set('page', String(params.page))
    if (params?.per_page) searchParams.set('per_page', String(params.per_page))
    if (params?.level) searchParams.set('level', params.level)
    if (params?.has_response !== undefined) searchParams.set('has_response', String(params.has_response))

    return request<{
      year: string
      page: number
      per_page: number
      total: number
      total_pages: number
      items: unknown[]
    }>(`/api/data/${year}/responses?${searchParams}`)
  },

  getSummary: (year: string) =>
    request<{
      year: string
      total_responses: number
      columns: string[]
      by_level?: Record<string, number>
      free_response_count?: number
    }>(`/api/data/${year}/summary`),

  syncToPocketbase: (year: string, options?: { force?: boolean; runAsync?: boolean }) =>
    request<{
      year: string
      status: string
      synced?: number
      total?: number
      existing_count?: number
      message?: string
      errors?: Array<{ respondent_id: string; error: string }>
      error_count?: number
    }>(`/api/data/${year}/sync-to-pocketbase`, {
      method: 'POST',
      body: { force: options?.force ?? false, run_async: options?.runAsync ?? false },
    }),

  getDataStatus: (year: string) =>
    request<{
      year: string
      has_data: boolean
      survey_response_count: number
      error?: string
    }>(`/api/data/${year}/data-status`),

  getUnifiedResponses: (year: string, params?: {
    page?: number
    per_page?: number
    level?: string
    is_minority?: boolean
    has_support?: boolean
    is_year_1?: boolean
    min_tenure?: number
    max_tenure?: number
    sort?: string
  }) => {
    const searchParams = new URLSearchParams()
    if (params?.page) searchParams.set('page', String(params.page))
    if (params?.per_page) searchParams.set('per_page', String(params.per_page))
    if (params?.level) searchParams.set('level', params.level)
    if (params?.is_minority !== undefined) searchParams.set('is_minority', String(params.is_minority))
    if (params?.has_support !== undefined) searchParams.set('has_support', String(params.has_support))
    if (params?.is_year_1 !== undefined) searchParams.set('is_year_1', String(params.is_year_1))
    if (params?.min_tenure !== undefined) searchParams.set('min_tenure', String(params.min_tenure))
    if (params?.max_tenure !== undefined) searchParams.set('max_tenure', String(params.max_tenure))
    if (params?.sort) searchParams.set('sort', params.sort)

    return request<PaginatedResponse<UnifiedResponse>>(
      `/api/data/${year}/unified-responses?${searchParams}`
    )
  },

  getColumnValues: (year: string, column: string) =>
    request<{
      year: string
      column: string
      values: (string | number | boolean | null)[]
    }>(`/api/data/${year}/column-values/${column}`),

  getSegments: (year: string) =>
    request<{
      year: string
      total: number
      segments: Array<{
        key: string
        name: string
        inverse_name: string
        count: number
        percentage: number
        inverse_count: number
        inverse_percentage: number
      }>
    }>(`/api/data/${year}/segments`),

  getSegmentStatistics: (year: string, segmentKey: string, inverse = false) => {
    const params = inverse ? '?inverse=true' : ''
    return request<{
      year: string
      segment: string
      segment_name: string
      inverse: boolean
      statistics: {
        total_responses: number
        weighted_averages: Record<string, number | null>
        level_counts: Record<string, number>
        question_totals: unknown[]
      }
    }>(`/api/data/${year}/segments/${segmentKey}/statistics${params}`)
  },

  // Get free responses with tags for the tagging interface (US1)
  getFreeResponsesWithTags: (year: string, params?: {
    page?: number
    per_page?: number
    question_type?: 'praise' | 'improvement'
    level?: string
    tag?: string
    has_override?: boolean
    min_stability?: number
    max_stability?: number
    has_keyword_mismatch?: boolean
    sort?: string
  }) => {
    const searchParams = new URLSearchParams()
    if (params?.page) searchParams.set('page', String(params.page))
    if (params?.per_page) searchParams.set('per_page', String(params.per_page))
    if (params?.question_type) searchParams.set('question_type', params.question_type)
    if (params?.level) searchParams.set('level', params.level)
    if (params?.tag) searchParams.set('tag', params.tag)
    if (params?.has_override !== undefined) searchParams.set('has_override', String(params.has_override))
    if (params?.min_stability !== undefined) searchParams.set('min_stability', String(params.min_stability))
    if (params?.max_stability !== undefined) searchParams.set('max_stability', String(params.max_stability))
    if (params?.has_keyword_mismatch !== undefined) searchParams.set('has_keyword_mismatch', String(params.has_keyword_mismatch))
    if (params?.sort) searchParams.set('sort', params.sort)

    return request<TaggableResponsePage>(
      `/api/data/${year}/free-responses-with-tags?${searchParams}`
    )
  },
}

// Tagging API
export const taggingApi = {
  start: (year: string, options?: {
    model?: string
    n_samples?: number
    threshold?: number
    use_fragments?: boolean
    use_batch_api?: boolean
    test_mode?: boolean
    test_size?: number
  }) =>
    request<{ job_id: string; year: string; status: string }>(`/api/tagging/${year}/start`, {
      method: 'POST',
      body: options || {},
    }),

  getResults: (year: string, params?: {
    page?: number
    per_page?: number
    level?: string
    tag?: string
    question?: string
  }) => {
    const searchParams = new URLSearchParams()
    if (params?.page) searchParams.set('page', String(params.page))
    if (params?.per_page) searchParams.set('per_page', String(params.per_page))
    if (params?.level) searchParams.set('level', params.level)
    if (params?.tag) searchParams.set('tag', params.tag)
    if (params?.question) searchParams.set('question', params.question)

    return request<{
      page: number
      perPage: number
      totalItems: number
      totalPages: number
      items: TaggingResult[]
    }>(`/api/tagging/${year}/results?${searchParams}`)
  },

  getDistribution: (year: string, level?: string, question?: string) => {
    const params = new URLSearchParams()
    if (level) params.set('level', level)
    if (question) params.set('question', question)
    const queryString = params.toString()
    return request<{
      year: string
      level: string | null
      total_tags: number
      unique_tags: number
      distribution: Array<{ tag: string; count: number; percentage: number }>
    }>(`/api/tagging/${year}/distribution${queryString ? `?${queryString}` : ''}`)
  },

  // Batch tagging API (FR-012)
  startBatch: (year: string, options?: { retag_existing?: boolean; model?: string }) =>
    request<BatchJobCreated>(`/api/tagging/${year}/batch`, {
      method: 'POST',
      body: {
        retag_existing: options?.retag_existing ?? false,
        model: options?.model,
      },
    }),

  getBatchStatus: (year: string, batchJobId: string) =>
    request<BatchJobStatus>(`/api/tagging/${year}/batch/${batchJobId}`),

  // Single response tagging (FR-013)
  tagSingle: (year: string, responseId: string, model?: string) => {
    const params = model ? `?model=${encodeURIComponent(model)}` : ''
    return request<SingleTaggingResult>(`/api/tagging/${year}/single/${responseId}${params}`, {
      method: 'POST',
    })
  },

  // Toggle a single tag on a response (US2 - inline editing)
  toggleTag: (year: string, responseId: string, tag: string, value: boolean) =>
    request<TagToggleResult>(`/api/tagging/${year}/responses/${responseId}/tags`, {
      method: 'PUT',
      body: { tag, value },
    }),

  // Review queue API (FR-017)
  getReviewQueue: (year: string, params?: {
    page?: number
    per_page?: number
    stability_threshold?: number
  }) => {
    const searchParams = new URLSearchParams()
    if (params?.page) searchParams.set('page', String(params.page))
    if (params?.per_page) searchParams.set('per_page', String(params.per_page))
    if (params?.stability_threshold) searchParams.set('stability_threshold', String(params.stability_threshold))

    return request<PaginatedResponse<ReviewQueueItem>>(`/api/tagging/${year}/review?${searchParams}`)
  },

  // Review actions (FR-018, FR-019, FR-020)
  approveReview: (year: string, responseId: string) =>
    request<{ status: string; response_id: string }>(`/api/tagging/${year}/review/${responseId}/approve`, {
      method: 'POST',
    }),

  hideFromReview: (year: string, responseId: string) =>
    request<{ status: string; response_id: string }>(`/api/tagging/${year}/review/${responseId}/hide`, {
      method: 'POST',
    }),

  modifyTags: (year: string, responseId: string, data: { tags: string[]; reason?: string }) =>
    request<TagModifyResult>(`/api/tagging/${year}/review/${responseId}/modify`, {
      method: 'PUT',
      body: data,
    }),

  // Tag-by-tag workflow
  getResponsesByTag: (year: string, tag: string, params?: {
    page?: number
    per_page?: number
    include_dismissed?: boolean
    sort?: string
  }) => {
    const searchParams = new URLSearchParams()
    if (params?.page) searchParams.set('page', String(params.page))
    if (params?.per_page) searchParams.set('per_page', String(params.per_page))
    if (params?.include_dismissed !== undefined) searchParams.set('include_dismissed', String(params.include_dismissed))
    if (params?.sort) searchParams.set('sort', params.sort)

    return request<PaginatedResponse<TaggedResponse>>(
      `/api/tagging/${year}/by-tag/${encodeURIComponent(tag)}?${searchParams}`
    )
  },

  dismissResponse: (year: string, responseId: string) =>
    request<{ status: string; response_id: string }>(`/api/tagging/${year}/responses/${responseId}/dismiss`, {
      method: 'POST',
    }),

  undismissResponse: (year: string, responseId: string) =>
    request<{ status: string; response_id: string }>(`/api/tagging/${year}/responses/${responseId}/undismiss`, {
      method: 'POST',
    }),

  getTagSummary: (year: string) =>
    request<{
      year: string
      tags: Array<{
        name: string
        pending: number
        dismissed: number
        total: number
      }>
    }>(`/api/tagging/${year}/tag-summary`),
}

// Tags API
export const tagsApi = {
  getTaxonomy: () =>
    request<{ tags: Array<{ name: string; keywords: string[] }> }>('/api/tags/taxonomy'),

  override: (year: string, responseId: string, data: {
    modified_tags: string[]
    reason?: string
  }) =>
    request<{
      status: string
      response_id: string
      original_tags: string[]
      modified_tags: string[]
    }>(`/api/tags/${year}/responses/${responseId}`, {
      method: 'PUT',
      body: data,
    }),

  getModifications: (year: string) =>
    request<{
      year: string
      total: number
      modifications: Array<{
        id: string
        response_id: string
        original_tags: string[]
        modified_tags: string[]
        reason: string | null
      }>
    }>(`/api/tags/${year}/modifications`),
}

// Clustering API
export const clusteringApi = {
  embed: (year: string, options?: {
    embed_model?: string
    question?: string
  }) =>
    request<{ job_id: string; year: string; status: string }>(`/api/clustering/${year}/embed`, {
      method: 'POST',
      body: options || {},
    }),

  start: (year: string, options?: {
    embed_model?: string
    min_cluster_size?: number
    question?: string
  }) =>
    request<{ job_id: string; year: string; status: string }>(`/api/clustering/${year}/start`, {
      method: 'POST',
      body: options || {},
    }),

  getResults: (year: string, params?: {
    question?: string
    cluster_id?: number
    page?: number
    per_page?: number
  }) => {
    const searchParams = new URLSearchParams()
    if (params?.page) searchParams.set('page', String(params.page))
    if (params?.per_page) searchParams.set('per_page', String(params.per_page))
    if (params?.question) searchParams.set('question', params.question)
    if (params?.cluster_id !== undefined) searchParams.set('cluster_id', String(params.cluster_id))

    return request<{
      page: number
      perPage: number
      totalItems: number
      totalPages: number
      items: ClusteringResult[]
    }>(`/api/clustering/${year}/results?${searchParams}`)
  },

  getCoordinates: (year: string, options?: { question?: string; includeTags?: boolean }) => {
    const params = new URLSearchParams()
    if (options?.question) params.set('question', options.question)
    if (options?.includeTags) params.set('include_tags', 'true')
    const queryString = params.toString()
    return request<{
      year: string
      question: string | null
      total: number
      coordinates: ClusterCoordinate[]
    }>(`/api/clustering/${year}/coordinates${queryString ? `?${queryString}` : ''}`)
  },

  getSummaries: (year: string, question?: string) => {
    const params = question ? `?question=${encodeURIComponent(question)}` : ''
    return request<{
      year: string
      question: string | null
      clusters: ClusterSummary[]
    }>(`/api/clustering/${year}/summaries${params}`)
  },

  summarize: (year: string, responseIds: string[], promptContext?: string) =>
    request<{
      summary: string
      key_points: string[]
      sentiment: 'positive' | 'negative' | 'mixed' | 'neutral'
      response_count: number
    }>(`/api/clustering/${year}/summarize`, {
      method: 'POST',
      body: {
        response_ids: responseIds,
        prompt_context: promptContext,
      },
    }),

  recluster: (year: string, responseIds: string[], minClusterSize?: number) =>
    request<ReclusterResult>(`/api/clustering/${year}/recluster`, {
      method: 'POST',
      body: {
        response_ids: responseIds,
        min_cluster_size: minClusterSize ?? 3,
      },
    }),

  getMetadata: (year: string) =>
    request<{ year: string; metadata: ClusterMetadata[] }>(`/api/clustering/${year}/metadata`),

  updateMetadata: (year: string, clusterId: number, data: { name?: string; description?: string }) =>
    request<ClusterMetadata>(`/api/clustering/${year}/metadata/${clusterId}`, {
      method: 'PUT',
      body: data,
    }),
}

// Artifacts API
export const artifactsApi = {
  list: (year: string) =>
    request<{
      year: string
      total: number
      artifacts: Array<{
        filename: string
        type: string
        size: number
        url: string
      }>
    }>(`/api/artifacts/${year}/list`),

  listAll: () =>
    request<{
      total: number
      artifacts: Array<{
        filename: string
        year: string | null
        type: string
        size: number
        url: string
      }>
    }>('/api/artifacts/list'),
}

// Jobs API
export const jobsApi = {
  getStatus: (jobId: string) =>
    request<Job>(`/api/jobs/${jobId}`),

  cancel: (jobId: string) =>
    request<{ status: string; job_id: string }>(`/api/jobs/${jobId}/cancel`, {
      method: 'POST',
    }),

  list: (year?: string) => {
    const params = year ? `?year=${year}` : ''
    return request<{ total: number; jobs: Job[] }>(`/api/jobs${params}`)
  },
}

// Batch Jobs API
export const batchJobsApi = {
  list: (year?: string, page = 1, perPage = 50) => {
    const params = new URLSearchParams()
    if (year) params.set('year', year)
    params.set('page', String(page))
    params.set('per_page', String(perPage))
    return request<{ items: BatchJob[]; totalItems: number; page: number; perPage: number }>(
      `/api/batch-jobs?${params}`
    )
  },

  get: (id: string) => request<BatchJob>(`/api/batch-jobs/${id}`),

  delete: (id: string) =>
    request<{ status: string }>(`/api/batch-jobs/${id}`, { method: 'DELETE' }),

  deleteGroup: (groupId: string) =>
    request<{ status: string; deleted: number }>(`/api/batch-jobs/groups/${groupId}`, { method: 'DELETE' }),

  cancel: (id: string) =>
    request<{ status: string; batch_id: string }>(`/api/batch-jobs/${id}/cancel`, { method: 'POST' }),

  poll: (id: string) =>
    request<BatchJob>(`/api/batch-jobs/${id}/poll`, { method: 'POST' }),

  retry: (id: string) =>
    request<{ status: string; new_batches: number }>(`/api/batch-jobs/${id}/retry`, { method: 'POST' }),

  clearAll: (year?: string) => {
    const params = new URLSearchParams()
    if (year) params.set('year', year)
    return request<{ status: string; deleted: number; cancelled: number }>(
      `/api/batch-jobs?${params}`, { method: 'DELETE' }
    )
  },
}

// Charts API
export const chartsApi = {
  generateAll: (year: string, includeTags = true) =>
    request<{
      year: string
      status: string
      charts_generated: number
      files: string[]
    }>(`/api/charts/${year}/generate-all`, {
      method: 'POST',
      body: { include_tags: includeTags },
    }),

  generateComparison: (years: string[]) =>
    request<{
      years: string[]
      status: string
      charts_generated: number
      files: string[]
    }>('/api/charts/compare/generate', {
      method: 'POST',
      body: { years },
    }),

  list: (year: string) =>
    request<{
      year: string
      total: number
      charts: Array<{ filename: string; size: number; url: string }>
    }>(`/api/charts/${year}/list`),

  listComparison: () =>
    request<{
      total: number
      charts: Array<{ filename: string; size: number; url: string }>
    }>('/api/charts/compare/list'),

  exportAll: (year: string, years?: string[]) =>
    request<{
      year: string
      status: string
      charts_generated: number
      files: string[]
    }>(`/api/charts/${year}/export-all`, {
      method: 'POST',
      body: { years },
    }),

  // New board presentation chart endpoints
  demographicComparison: (year: string, data: DemographicComparisonRequest) =>
    request<ChartResult>(`/api/charts/${year}/demographic-comparison`, {
      method: 'POST',
      body: data,
    }),

  trendComparison: (data: TrendComparisonRequest) =>
    request<ChartResult>('/api/charts/trend-comparison', {
      method: 'POST',
      body: data,
    }),

  sentiment: (year: string, data?: SentimentRequest) =>
    request<SentimentChartResult>(`/api/charts/${year}/sentiment`, {
      method: 'POST',
      body: data || {},
    }),
}

// Config API
export const configApi = {
  getQuestions: () =>
    request<{
      questions: string[]
      levels: string[]
    }>('/api/config/questions'),

  getTaxonomy: () =>
    request<{
      tags: Array<{ name: string; keywords: string[] }>
    }>('/api/config/taxonomy'),

  getScales: () =>
    request<{
      scales: Record<string, string[]>
    }>('/api/config/scales'),

  getFull: () =>
    request<{
      questions: string[]
      levels: string[]
      scales: Record<string, string[]>
      taxonomy: Array<{ name: string; keywords: string[] }>
    }>('/api/config'),
}

// Types
export interface YearStatus {
  year: string
  loaded: boolean
  row_count: number
  tagging_count: number
  tagging_complete: boolean
  clustering_count: number
  clustering_complete: boolean
  embeddings_complete: boolean
}

export interface Job {
  id: string
  job_type: string
  year: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  progress: number
  total_items: number
  processed_items: number
  error_message: string | null
  started_at: string | null
  completed_at: string | null
  metadata: Record<string, unknown>
}

export interface TaggingResult {
  id: string
  year: string
  response_id: string
  question: string
  level: string
  response_text: string
  llm_tags: string[]
  tag_votes: Record<string, number>
  keywords_found: Record<string, string[]>
  model_used: string
  n_samples: number
  threshold: number
}

export interface ClusteringResult {
  id: string
  year: string
  response_id: string
  question: string
  umap_x: number
  umap_y: number
  cluster_id: number
  embed_model: string
}

export interface ClusterSummary {
  id: string
  year: string
  question: string
  cluster_id: number
  size: number
  sample_responses: string[]
  tag_distribution: Record<string, { count: number; percentage: number }>
  centroid_x: number
  centroid_y: number
}

export interface ClusterCoordinate {
  response_id: string
  x: number
  y: number
  cluster_id: number
  level?: string
  tags?: string[]
  response_text?: string
  question?: string
}

export interface ClusterMetadata {
  id: string
  year: string
  cluster_id: number
  name: string
  description: string
}

export interface ReclusterResult {
  coordinates: ClusterCoordinate[]
  clusters: ClusterSummary[]
  total: number
  cluster_count: number
  noise_count: number
}

// New types for data import and table view
export interface ImportResult {
  year: string
  total_responses: number
  free_responses_extracted: number
  replaced_existing: boolean
}

export interface PaginatedResponse<T> {
  items: T[]
  page: number
  perPage: number
  totalItems: number
  totalPages: number
}

export interface SurveyResponse {
  id: string
  year: string
  respondent_id: string
  school_level: string | null
  submission_method: string | null
  n_parents_represented: number | null
  demographics: {
    minority: boolean
    support: boolean
    tenure_years: number | null
    year1_family: boolean | null
  } | null
  satisfaction_scores: Record<string, Record<string, string>> | null
  imported_at: string
}

export interface FreeResponse {
  id: string
  year: string
  response_id: string
  question: string
  question_type: 'praise' | 'improvement'
  level: string | null
  response_text: string
}

// Unified response with all survey data + free responses combined (no tags)
export interface UnifiedResponse {
  id: string
  respondent_id: string
  year: string
  level: string | null
  // Demographics
  is_minority: boolean | null
  has_support: boolean | null
  years_at_gvca: number | null
  is_year_1: boolean | null
  n_parents: number
  // Satisfaction Q1-Q7 × 3 levels
  satisfaction: {
    Q1: { Grammar: string | null; Middle: string | null; High: string | null }
    Q2: { Grammar: string | null; Middle: string | null; High: string | null }
    Q3: { Grammar: string | null; Middle: string | null; High: string | null }
    Q4: { Grammar: string | null; Middle: string | null; High: string | null }
    Q5: { Grammar: string | null; Middle: string | null; High: string | null }
    Q6: { Grammar: string | null; Middle: string | null; High: string | null }
    Q7: { Grammar: string | null; Middle: string | null; High: string | null }
  }
  // Free responses embedded (no tags - handled in Tagging page)
  free_responses: {
    praise: { Grammar: string | null; Middle: string | null; High: string | null; Generic: string | null }
    improvement: { Grammar: string | null; Middle: string | null; High: string | null; Generic: string | null }
  }
}

// Chart request/response types
export interface DemographicComparisonRequest {
  segment_a: string
  segment_b: string
  questions?: string[]
  export_png?: boolean
}

export interface TrendComparisonRequest {
  years: string[]
  school_level?: string
  questions?: string[]
  export_png?: boolean
}

export interface SentimentRequest {
  school_level?: string
  demographic?: string
  export_png?: boolean
}

export interface ChartResult {
  data: Array<{
    question: string
    segment_a_avg: number
    segment_b_avg: number
    difference: number
  } | {
    year: string
    question?: string
    average: number
  }>
  file_path?: string
}

export interface SentimentChartResult {
  data: Array<{
    tag: string
    positive_count: number
    negative_count: number
    net_sentiment: number
  }>
  file_path?: string
}

// Batch tagging types
export interface BatchJobCreated {
  job_id: string
  internal_job_id?: string
  job_type: string
  status: string
  total_items: number
  message: string
}

export interface BatchJobStatus {
  job_id: string
  year: string
  status: string
  total_items: number
  processed_items: number
  failed_items: number
  started_at: string
  completed_at: string | null
  openai_status?: {
    batch_id: string
    status: string
    request_counts?: {
      total: number
      completed: number
      failed: number
    }
  }
}

export interface BatchJob {
  id: string
  job_type: string
  year: string
  status: string
  total_items: number
  processed_items: number
  failed_items: number
  openai_batch_id: string
  input_file_id: string
  output_file_id?: string
  error_file_id?: string
  error_message?: string
  model_used?: string
  batch_group_id?: string
  estimated_tokens?: number
  started_at: string
  completed_at: string | null
  created: string
  updated: string
}

export interface SingleTaggingResult {
  response_id: string
  llm_tags: string[]
  stability_score: number
  keyword_mismatches: Array<{ tag: string; keywords: string[] }>
  review_status: string | null
}

// Review queue types
export interface ReviewQueueItem {
  id: string
  year: string
  response_id: string
  question: string
  level: string | null
  response_text: string
  llm_tags: string[]
  tag_votes: Record<string, number>
  stability_score: number
  keyword_mismatches: Array<{ tag: string; keywords: string[] }>
  review_status: string | null
  review_flags: string[]
  review_priority: number
}

export interface TagModifyResult {
  response_id: string
  original_tags: string[]
  modified_tags: string[]
  reason: string | null
  status: string
}

export interface TagToggleResult {
  response_id: string
  tag: string
  value: boolean
  tags: string[]
  original_tags: string[]
}

// ========== Tagging Interface Types ==========

export interface KeywordMismatch {
  tag: string
  keywords: string[]
}

export interface TaggableResponse {
  id: string | null  // tagging_results record ID (null if not tagged yet)
  response_id: string
  respondent_id: string
  question: string  // Q8 or Q9
  question_type: 'praise' | 'improvement'
  level: string  // Grammar/Middle/High/Generic
  response_text: string
  // Tags (from tagging_results, merged with overrides)
  tags: string[]
  original_tags: string[]
  has_override: boolean
  // Quality indicators
  stability_score: number | null
  tag_votes: Record<string, number>
  keyword_mismatches: KeywordMismatch[]
  // Status
  dismissed: boolean
  dismissed_at: string | null
}

export interface TaggableResponsePage {
  items: TaggableResponse[]
  page: number
  perPage: number
  totalItems: number
  totalPages: number
}

// Tagged response for tag-by-tag workflow
export interface TaggedResponse {
  id: string
  year: string
  response_id: string
  question: string
  level: string | null
  response_text: string
  // Tagging data
  llm_tags: string[]
  tag_votes: Record<string, number>
  stability_score: number
  keywords_found: Record<string, string[]>
  model_used: string
  n_samples: number
  // Dismiss workflow
  dismissed: boolean
  dismissed_at: string | null
  // Review status
  review_status: string | null
}

// WebSocket helper for job updates
export function subscribeToJob(
  jobId: string,
  onUpdate: (data: Job) => void,
  onError?: (error: Event) => void
): () => void {
  const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/jobs/ws/${jobId}`
  const ws = new WebSocket(wsUrl)

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data)
    if (data.type !== 'heartbeat') {
      onUpdate(data)
    }
  }

  ws.onerror = (error) => {
    console.error('WebSocket error:', error)
    if (onError) {
      onError(error)
    }
  }

  return () => {
    ws.close()
  }
}
