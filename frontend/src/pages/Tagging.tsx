import { useState, useCallback } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { taggingApi, type TaggableResponsePage } from '@/api/client'
import { useAppStore } from '@/stores/app'
import { Card, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { TaggingTable } from '@/components/tagging/TaggingTable'
import {
  RefreshCw,
} from 'lucide-react'

export default function Tagging() {
  const { selectedYear } = useAppStore()
  const queryClient = useQueryClient()
  const [loadingTags, setLoadingTags] = useState<Set<string>>(new Set())

  // Start batch tagging mutation
  const batchMutation = useMutation({
    mutationFn: (retagExisting: boolean) =>
      taggingApi.startBatch(selectedYear!, { retag_existing: retagExisting }),
  })

  // Tag toggle mutation for table view with optimistic updates (US2)
  const tagToggleMutation = useMutation({
    mutationFn: async ({ responseId, tag, currentValue }: { responseId: string; tag: string; currentValue: boolean }) => {
      return taggingApi.toggleTag(selectedYear!, responseId, tag, !currentValue)
    },
    onMutate: async ({ responseId, tag, currentValue }) => {
      // Add to loading state
      setLoadingTags((prev) => new Set(prev).add(`${responseId}:${tag}`))

      // Cancel any outgoing refetches
      await queryClient.cancelQueries({ queryKey: ['free-responses-with-tags'] })

      // Snapshot the previous value
      const previousData = queryClient.getQueryData(['free-responses-with-tags', selectedYear]) as TaggableResponsePage | undefined

      // Optimistically update the cache
      if (previousData) {
        queryClient.setQueryData(['free-responses-with-tags', selectedYear], (old: TaggableResponsePage | undefined) => {
          if (!old) return old
          return {
            ...old,
            items: old.items.map((item) => {
              if (item.response_id === responseId) {
                const newTags = currentValue
                  ? item.tags.filter((t) => t !== tag)
                  : [...item.tags, tag]
                return { ...item, tags: newTags, has_override: true }
              }
              return item
            }),
          }
        })
      }

      return { previousData }
    },
    onError: (_err, { responseId, tag }, context) => {
      // Rollback on error
      if (context?.previousData) {
        queryClient.setQueryData(['free-responses-with-tags', selectedYear], context.previousData)
      }
      // Remove from loading state
      setLoadingTags((prev) => {
        const next = new Set(prev)
        next.delete(`${responseId}:${tag}`)
        return next
      })
    },
    onSuccess: (_data, { responseId, tag }) => {
      // Remove from loading state
      setLoadingTags((prev) => {
        const next = new Set(prev)
        next.delete(`${responseId}:${tag}`)
        return next
      })
    },
    onSettled: () => {
      // Always refetch to ensure consistency
      queryClient.invalidateQueries({ queryKey: ['free-responses-with-tags'] })
    },
  })

  const handleTagToggle = useCallback((responseId: string, tag: string, currentValue: boolean) => {
    // Prevent race condition: don't start a new mutation if one is already in progress for this tag
    const tagKey = `${responseId}:${tag}`
    if (loadingTags.has(tagKey)) {
      return
    }
    tagToggleMutation.mutate({ responseId, tag, currentValue })
  }, [tagToggleMutation, loadingTags])

  if (!selectedYear) {
    return (
      <div className="flex items-center justify-center h-[50vh]">
        <Card className="w-96">
          <CardHeader>
            <CardTitle>No Year Selected</CardTitle>
            <CardDescription>
              Select a survey year from the dropdown to view tagging results.
            </CardDescription>
          </CardHeader>
        </Card>
      </div>
    )
  }

  return (
    <div className="h-[calc(100vh-3.5rem)] overflow-hidden flex flex-col">
      {/* Table View */}
      <div className="flex-1 overflow-auto p-6">
        <TaggingTable
          onTagToggle={handleTagToggle}
          loadingTags={loadingTags}
          batchTagAction={
            <Button
              size="sm"
              onClick={() => batchMutation.mutate(false)}
              disabled={batchMutation.isPending}
            >
              {batchMutation.isPending ? (
                <>
                  <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  Starting...
                </>
              ) : (
                'Start Batch Tagging'
              )}
            </Button>
          }
        />
      </div>
    </div>
  )
}
