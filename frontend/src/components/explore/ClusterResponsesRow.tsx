import { useState, useCallback } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import type { ClusterCoordinate } from '@/api/client'
import { taggingApi } from '@/api/client'
import { useAppStore } from '@/stores/app'
import { Badge } from '@/components/ui/badge'
import { TagCell } from '@/components/tagging/TagCell'

interface ClusterResponsesRowProps {
  year: string
  responses: ClusterCoordinate[]
  taxonomy: Array<{ name: string; keywords: string[] }>
}

function questionLabel(question?: string): string {
  if (!question) return ''
  if (question.startsWith('What makes')) return 'Q8'
  if (question.startsWith('Please provide')) return 'Q9'
  return question.slice(0, 20)
}

export default function ClusterResponsesRow({
  year,
  responses,
  taxonomy,
}: ClusterResponsesRowProps) {
  const queryClient = useQueryClient()
  const selectedYear = useAppStore((s) => s.selectedYear)
  const [loadingTags, setLoadingTags] = useState<Set<string>>(new Set())

  const tagToggleMutation = useMutation({
    mutationFn: async ({
      responseId,
      tag,
      currentValue,
    }: {
      responseId: string
      tag: string
      currentValue: boolean
    }) => {
      return taggingApi.toggleTag(year, responseId, tag, !currentValue)
    },
    onMutate: async ({ responseId, tag }) => {
      setLoadingTags((prev) => new Set(prev).add(`${responseId}:${tag}`))
    },
    onSuccess: (_data, { responseId, tag }) => {
      setLoadingTags((prev) => {
        const next = new Set(prev)
        next.delete(`${responseId}:${tag}`)
        return next
      })
    },
    onError: (_err, { responseId, tag }) => {
      setLoadingTags((prev) => {
        const next = new Set(prev)
        next.delete(`${responseId}:${tag}`)
        return next
      })
    },
    onSettled: () => {
      // Refresh coordinates to pick up tag changes
      queryClient.invalidateQueries({ queryKey: ['cluster-coordinates', selectedYear] })
    },
  })

  const handleTagToggle = useCallback(
    (responseId: string, tagName: string, currentValue: boolean) => {
      const tagKey = `${responseId}:${tagName}`
      if (loadingTags.has(tagKey)) return
      tagToggleMutation.mutate({ responseId, tag: tagName, currentValue })
    },
    [tagToggleMutation, loadingTags],
  )

  const tagNames = taxonomy.map((t) => t.name)

  return (
    <div className="bg-muted/20 border-t">
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead className="bg-muted/30 border-b">
            <tr>
              <th className="text-left px-3 py-1.5 w-20">Level</th>
              <th className="text-left px-3 py-1.5 w-16">Q</th>
              <th className="text-left px-3 py-1.5">Response</th>
              {tagNames.map((tag) => (
                <th key={tag} className="px-2 py-1.5 text-center w-16 whitespace-nowrap">
                  <span className="writing-mode-vertical" title={tag}>
                    {tag.length > 12 ? tag.slice(0, 11) + '\u2026' : tag}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {responses.map((r) => (
              <tr key={r.response_id} className="border-b last:border-b-0 hover:bg-muted/30">
                <td className="px-3 py-1.5">
                  {r.level && <Badge variant="outline" className="text-xs">{r.level}</Badge>}
                </td>
                <td className="px-3 py-1.5 text-muted-foreground">
                  {questionLabel(r.question)}
                </td>
                <td className="px-3 py-1.5 max-w-md">{r.response_text}</td>
                {tagNames.map((tag) => {
                  const isTagged = r.tags?.includes(tag) ?? false
                  const tagKey = `${r.response_id}:${tag}`
                  return (
                    <td key={tag} className="px-2 py-1.5 text-center">
                      <TagCell
                        responseId={r.response_id}
                        tagName={tag}
                        isTagged={isTagged}
                        isLoading={loadingTags.has(tagKey)}
                        onToggle={handleTagToggle}
                      />
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
