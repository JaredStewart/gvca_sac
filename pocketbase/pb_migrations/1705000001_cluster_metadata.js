/// <reference path="../pb_data/types.d.ts" />

// Migration for cluster_metadata collection — user-editable names and descriptions for clusters

migrate((db) => {
  const dao = new Dao(db);

  const clusterMetadata = new Collection({
    name: 'cluster_metadata',
    type: 'base',
    listRule: '',
    viewRule: '',
    createRule: '',
    updateRule: '',
    deleteRule: '',
    schema: [
      { name: 'year', type: 'text', required: true },
      { name: 'cluster_id', type: 'number', required: false },
      { name: 'name', type: 'text', required: false },
      { name: 'description', type: 'text', required: false },
    ],
    indexes: [
      'CREATE UNIQUE INDEX idx_cluster_meta_year_id ON cluster_metadata (year, cluster_id)',
    ],
  });
  dao.saveCollection(clusterMetadata);

}, (db) => {
  const dao = new Dao(db);
  try {
    const collection = dao.findCollectionByNameOrId('cluster_metadata');
    if (collection) {
      dao.deleteCollection(collection);
    }
  } catch (e) {
    // Collection might not exist
  }
});
