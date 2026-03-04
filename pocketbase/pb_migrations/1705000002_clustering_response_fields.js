/// <reference path="../pb_data/types.d.ts" />

// Add response_text and level fields to clustering_results
// so coordinates can be served without joining against tagging_results.

migrate((db) => {
  const dao = new Dao(db);
  const collection = dao.findCollectionByNameOrId('clustering_results');

  // Add response_text field
  collection.schema.addField(new SchemaField({
    name: 'response_text',
    type: 'text',
    required: false,
  }));

  // Add level field
  collection.schema.addField(new SchemaField({
    name: 'level',
    type: 'text',
    required: false,
  }));

  dao.saveCollection(collection);

}, (db) => {
  const dao = new Dao(db);
  const collection = dao.findCollectionByNameOrId('clustering_results');

  collection.schema.removeField('response_text');
  collection.schema.removeField('level');

  dao.saveCollection(collection);
});
