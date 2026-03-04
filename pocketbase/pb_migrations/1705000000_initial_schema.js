/// <reference path="../pb_data/types.d.ts" />

// Initial schema migration for GVCA SAC collections

migrate((db) => {
  const dao = new Dao(db);

  // tagging_results - LLM tagging output
  const taggingResults = new Collection({
    name: 'tagging_results',
    type: 'base',
    listRule: '',
    viewRule: '',
    createRule: '',
    updateRule: '',
    deleteRule: '',
    schema: [
      { name: 'year', type: 'text', required: true },
      { name: 'response_id', type: 'text', required: true },
      { name: 'question', type: 'text', required: true },
      { name: 'level', type: 'text', required: false },
      { name: 'response_text', type: 'text', required: true },
      { name: 'llm_tags', type: 'json', required: false, options: { maxSize: 2000000 } },
      { name: 'tag_votes', type: 'json', required: false, options: { maxSize: 2000000 } },
      { name: 'keywords_found', type: 'json', required: false, options: { maxSize: 2000000 } },
      { name: 'model_used', type: 'text', required: false },
      { name: 'n_samples', type: 'number', required: false },
      { name: 'threshold', type: 'number', required: false },
      // New fields for stability scoring and review workflow
      { name: 'stability_score', type: 'number', required: false },  // 0.0-1.0 agreement across AI runs
      { name: 'keyword_mismatches', type: 'json', required: false, options: { maxSize: 2000000 } },  // [{tag: "X", keywords: ["a","b"]}]
      { name: 'review_status', type: 'text', required: false },  // "pending", "approved", "hidden"
      { name: 'batch_id', type: 'text', required: false },  // Reference to batch_jobs.id
      // Dismiss workflow fields
      { name: 'dismissed', type: 'bool', required: false },
      { name: 'dismissed_at', type: 'date', required: false },
    ],
    indexes: [
      'CREATE INDEX idx_tagging_year ON tagging_results (year)',
      'CREATE INDEX idx_tagging_response ON tagging_results (response_id)',
      'CREATE INDEX idx_tagging_review_status ON tagging_results (review_status)',
      'CREATE INDEX idx_tagging_stability ON tagging_results (stability_score)',
    ],
  });
  dao.saveCollection(taggingResults);

  // tag_overrides - User modifications to tags
  const tagOverrides = new Collection({
    name: 'tag_overrides',
    type: 'base',
    listRule: '',
    viewRule: '',
    createRule: '',
    updateRule: '',
    deleteRule: '',
    schema: [
      { name: 'year', type: 'text', required: true },
      { name: 'response_id', type: 'text', required: true },
      { name: 'question', type: 'text', required: false },
      { name: 'original_tags', type: 'json', required: false, options: { maxSize: 2000000 } },
      { name: 'modified_tags', type: 'json', required: false, options: { maxSize: 2000000 } },
      { name: 'reason', type: 'text', required: false },
      { name: 'modified_at', type: 'date', required: false },  // When override was applied
    ],
    indexes: [
      'CREATE INDEX idx_override_year ON tag_overrides (year)',
      'CREATE INDEX idx_override_response ON tag_overrides (response_id)',
    ],
  });
  dao.saveCollection(tagOverrides);

  // clustering_results - Embeddings and coordinates
  const clusteringResults = new Collection({
    name: 'clustering_results',
    type: 'base',
    listRule: '',
    viewRule: '',
    createRule: '',
    updateRule: '',
    deleteRule: '',
    schema: [
      { name: 'year', type: 'text', required: true },
      { name: 'response_id', type: 'text', required: true },
      { name: 'question', type: 'text', required: true },
      { name: 'umap_x', type: 'number', required: false },
      { name: 'umap_y', type: 'number', required: false },
      { name: 'cluster_id', type: 'number', required: false },
      { name: 'embed_model', type: 'text', required: false },
    ],
    indexes: [
      'CREATE INDEX idx_clustering_year ON clustering_results (year)',
      'CREATE INDEX idx_clustering_cluster ON clustering_results (cluster_id)',
    ],
  });
  dao.saveCollection(clusteringResults);

  // cluster_summaries - Cluster metadata
  // Note: cluster_id is not required because PocketBase treats 0 as "missing" for required number fields
  const clusterSummaries = new Collection({
    name: 'cluster_summaries',
    type: 'base',
    listRule: '',
    viewRule: '',
    createRule: '',
    updateRule: '',
    deleteRule: '',
    schema: [
      { name: 'year', type: 'text', required: true },
      { name: 'question', type: 'text', required: true },
      { name: 'cluster_id', type: 'number', required: false },
      { name: 'size', type: 'number', required: false },
      { name: 'sample_responses', type: 'json', required: false, options: { maxSize: 2000000 } },
      { name: 'tag_distribution', type: 'json', required: false, options: { maxSize: 2000000 } },
      { name: 'centroid_x', type: 'number', required: false },
      { name: 'centroid_y', type: 'number', required: false },
    ],
    indexes: [
      'CREATE INDEX idx_cluster_summary_year ON cluster_summaries (year)',
    ],
  });
  dao.saveCollection(clusterSummaries);

  // segment_summaries - Pre-computed segment stats (placeholder: not yet written to by backend)
  const segmentSummaries = new Collection({
    name: 'segment_summaries',
    type: 'base',
    listRule: '',
    viewRule: '',
    createRule: '',
    updateRule: '',
    deleteRule: '',
    schema: [
      { name: 'year', type: 'text', required: true },
      { name: 'segment_name', type: 'text', required: true },
      { name: 'tag_counts', type: 'json', required: false, options: { maxSize: 2000000 } },
      { name: 'total_responses', type: 'number', required: false },
      { name: 'sample_responses', type: 'json', required: false, options: { maxSize: 2000000 } },
    ],
    indexes: [
      'CREATE INDEX idx_segment_year ON segment_summaries (year)',
    ],
  });
  dao.saveCollection(segmentSummaries);

  // batch_jobs - Job tracking and costs
  const batchJobs = new Collection({
    name: 'batch_jobs',
    type: 'base',
    listRule: '',
    viewRule: '',
    createRule: '',
    updateRule: '',
    deleteRule: '',
    schema: [
      { name: 'job_type', type: 'text', required: true },  // "tagging_batch", "clustering", "import"
      { name: 'year', type: 'text', required: true },
      { name: 'status', type: 'text', required: true },  // validating, failed, in_progress, finalizing, completed, expired
      { name: 'progress', type: 'number', required: false },
      { name: 'total_items', type: 'number', required: false },
      { name: 'processed_items', type: 'number', required: false },
      { name: 'failed_items', type: 'number', required: false },  // Count of failed requests in batch
      { name: 'openai_batch_id', type: 'text', required: false },
      { name: 'input_file_id', type: 'text', required: false },  // OpenAI Files API ID for input JSONL
      { name: 'output_file_id', type: 'text', required: false },  // OpenAI Files API ID for results
      { name: 'error_file_id', type: 'text', required: false },  // OpenAI Files API ID for errors
      { name: 'estimated_cost', type: 'number', required: false },
      { name: 'model_used', type: 'text', required: false },
      { name: 'batch_group_id', type: 'text', required: false },
      { name: 'estimated_tokens', type: 'number', required: false },
      { name: 'error_message', type: 'text', required: false },
      { name: 'started_at', type: 'text', required: false },
      { name: 'completed_at', type: 'text', required: false },
      { name: 'metadata', type: 'json', required: false, options: { maxSize: 2000000 } },
    ],
    indexes: [
      'CREATE INDEX idx_job_type ON batch_jobs (job_type)',
      'CREATE INDEX idx_job_status ON batch_jobs (status)',
      'CREATE INDEX idx_job_year ON batch_jobs (year)',
      'CREATE INDEX idx_job_openai_batch ON batch_jobs (openai_batch_id)',
    ],
  });
  dao.saveCollection(batchJobs);

}, (db) => {
  // Rollback
  const dao = new Dao(db);
  const collections = ['batch_jobs', 'segment_summaries', 'cluster_summaries', 'clustering_results', 'tag_overrides', 'tagging_results'];

  for (const name of collections) {
    try {
      const collection = dao.findCollectionByNameOrId(name);
      if (collection) {
        dao.deleteCollection(collection);
      }
    } catch (e) {
      // Collection might not exist
    }
  }
});
