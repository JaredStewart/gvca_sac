/// <reference path="../pb_data/types.d.ts" />

// Migration to add survey_responses and free_responses collections for storing survey data

migrate((db) => {
  const dao = new Dao(db);

  // survey_responses - Normalized survey data with demographics and satisfaction scores
  const surveyResponses = new Collection({
    name: 'survey_responses',
    type: 'base',
    listRule: '',
    viewRule: '',
    createRule: '',
    updateRule: '',
    deleteRule: '',
    schema: [
      { name: 'year', type: 'text', required: true },
      { name: 'respondent_id', type: 'text', required: true },
      { name: 'school_level', type: 'text', required: false },
      { name: 'submission_method', type: 'text', required: false },
      { name: 'n_parents_represented', type: 'number', required: false },
      { name: 'demographics', type: 'json', required: false, options: { maxSize: 2000000 } },
      { name: 'satisfaction_scores', type: 'json', required: false, options: { maxSize: 2000000 } },
      { name: 'imported_at', type: 'date', required: true },
    ],
    indexes: [
      'CREATE INDEX idx_survey_responses_year ON survey_responses (year)',
      'CREATE INDEX idx_survey_responses_respondent ON survey_responses (respondent_id)',
      'CREATE INDEX idx_survey_responses_school_level ON survey_responses (school_level)',
    ],
  });
  dao.saveCollection(surveyResponses);

  // free_responses - Individual free-text answers extracted from survey responses
  const freeResponses = new Collection({
    name: 'free_responses',
    type: 'base',
    listRule: '',
    viewRule: '',
    createRule: '',
    updateRule: '',
    deleteRule: '',
    schema: [
      { name: 'year', type: 'text', required: true },
      { name: 'response_id', type: 'text', required: true },
      { name: 'survey_response_id', type: 'text', required: false },
      { name: 'question', type: 'text', required: true },
      { name: 'question_type', type: 'text', required: true },
      { name: 'level', type: 'text', required: false },
      { name: 'response_text', type: 'text', required: true },
    ],
    indexes: [
      'CREATE INDEX idx_free_responses_year ON free_responses (year)',
      'CREATE UNIQUE INDEX idx_free_responses_response_id ON free_responses (response_id)',
      'CREATE INDEX idx_free_responses_question_type ON free_responses (question_type)',
      'CREATE INDEX idx_free_responses_level ON free_responses (level)',
    ],
  });
  dao.saveCollection(freeResponses);

}, (db) => {
  // Rollback
  const dao = new Dao(db);
  const collections = ['free_responses', 'survey_responses'];

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
