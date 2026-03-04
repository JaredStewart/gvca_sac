import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import type { ClusterCoordinate, ReclusterResult } from '@/api/client'
import { clusteringApi } from '@/api/client'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Loader2, Sparkles, X, RefreshCw } from 'lucide-react'

interface SelectionSummaryPanelProps {
  year: string
  selectedResponses: ClusterCoordinate[]
  onClose: () => void
  onRecluster?: (result: ReclusterResult) => void
}

export default function SelectionSummaryPanel({
  year,
  selectedResponses,
  onClose,
  onRecluster,
}: SelectionSummaryPanelProps) {
  const [promptContext, setPromptContext] = useState('')

  const summarizeMutation = useMutation({
    mutationFn: () => clusteringApi.summarize(
      year,
      selectedResponses.map(r => r.response_id),
      promptContext || undefined,
    ),
  })

  const reclusterMutation = useMutation({
    mutationFn: () => clusteringApi.recluster(
      year,
      selectedResponses.map(r => r.response_id),
    ),
    onSuccess: (result) => {
      onRecluster?.(result)
    },
  })

  if (selectedResponses.length === 0) return null

  // Aggregate tag counts
  const tagCounts: Record<string, number> = {}
  const levelCounts: Record<string, number> = {}
  const clusterCounts: Record<number, number> = {}

  selectedResponses.forEach(r => {
    if (r.tags) {
      r.tags.forEach(tag => {
        tagCounts[tag] = (tagCounts[tag] || 0) + 1
      })
    }
    if (r.level) {
      levelCounts[r.level] = (levelCounts[r.level] || 0) + 1
    }
    clusterCounts[r.cluster_id] = (clusterCounts[r.cluster_id] || 0) + 1
  })

  const sortedTags = Object.entries(tagCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'positive': return 'bg-green-100 text-green-800 border-green-300'
      case 'negative': return 'bg-red-100 text-red-800 border-red-300'
      case 'mixed': return 'bg-yellow-100 text-yellow-800 border-yellow-300'
      default: return 'bg-gray-100 text-gray-800 border-gray-300'
    }
  }

  return (
    <Card className="w-96">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm">Selection Summary</CardTitle>
          <button
            onClick={onClose}
            className="p-1 hover:bg-muted rounded"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <CardDescription>
          {selectedResponses.length} responses selected
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Level breakdown */}
        <div>
          <div className="text-xs font-medium text-muted-foreground mb-1">By Level</div>
          <div className="flex flex-wrap gap-1">
            {Object.entries(levelCounts).map(([level, count]) => (
              <Badge key={level} variant="outline" className="text-xs">
                {level}: {count}
              </Badge>
            ))}
          </div>
        </div>

        {/* Cluster breakdown */}
        <div>
          <div className="text-xs font-medium text-muted-foreground mb-1">By Cluster</div>
          <div className="flex flex-wrap gap-1">
            {Object.entries(clusterCounts)
              .sort((a, b) => parseInt(a[0]) - parseInt(b[0]))
              .map(([cluster, count]) => (
                <Badge key={cluster} variant="outline" className="text-xs">
                  {cluster === '-1' ? 'Noise' : `C${cluster}`}: {count}
                </Badge>
              ))}
          </div>
        </div>

        {/* Top tags */}
        {sortedTags.length > 0 && (
          <div>
            <div className="text-xs font-medium text-muted-foreground mb-1">Top Tags</div>
            <div className="flex flex-wrap gap-1">
              {sortedTags.map(([tag, count]) => (
                <Badge key={tag} variant="secondary" className="text-xs">
                  {tag} ({count})
                </Badge>
              ))}
            </div>
          </div>
        )}

        {/* Re-cluster Selection */}
        {onRecluster && (
          <div className="pt-2 border-t">
            <Button
              onClick={() => reclusterMutation.mutate()}
              disabled={reclusterMutation.isPending || selectedResponses.length < 5}
              className="w-full"
              size="sm"
              variant="outline"
            >
              {reclusterMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Re-clustering...
                </>
              ) : (
                <>
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Re-cluster Selection
                </>
              )}
            </Button>
            {selectedResponses.length < 5 && (
              <p className="text-xs text-muted-foreground mt-1">
                Need at least 5 responses to re-cluster
              </p>
            )}
            {reclusterMutation.error && (
              <p className="text-xs text-red-500 mt-1">
                {(reclusterMutation.error as Error).message}
              </p>
            )}
          </div>
        )}

        {/* AI Summarization */}
        <div className="pt-2 border-t">
          <div className="text-xs font-medium text-muted-foreground mb-2">AI Summary</div>

          <div className="space-y-2">
            <div>
              <Label htmlFor="context" className="text-xs">Focus (optional)</Label>
              <Input
                id="context"
                placeholder="e.g., communication issues"
                value={promptContext}
                onChange={(e) => setPromptContext(e.target.value)}
                className="h-8 text-sm"
              />
            </div>

            <Button
              onClick={() => summarizeMutation.mutate()}
              disabled={summarizeMutation.isPending}
              className="w-full"
              size="sm"
            >
              {summarizeMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Summarizing...
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4 mr-2" />
                  Summarize Selection
                </>
              )}
            </Button>
          </div>

          {/* Summary Result */}
          {summarizeMutation.data && (
            <div className="mt-3 space-y-2">
              <div className={`px-2 py-1 rounded text-xs border ${getSentimentColor(summarizeMutation.data.sentiment)}`}>
                Sentiment: {summarizeMutation.data.sentiment}
              </div>

              <div className="text-sm">{summarizeMutation.data.summary}</div>

              {summarizeMutation.data.key_points.length > 0 && (
                <div>
                  <div className="text-xs font-medium text-muted-foreground mb-1">Key Points</div>
                  <ul className="text-xs space-y-1 list-disc list-inside">
                    {summarizeMutation.data.key_points.map((point, idx) => (
                      <li key={idx}>{point}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {summarizeMutation.error && (
            <div className="mt-2 text-xs text-red-500">
              Error generating summary. Please try again.
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
