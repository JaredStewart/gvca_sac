import type { ClusterCoordinate } from '@/api/client'
import { Badge } from '@/components/ui/badge'
import { CLUSTER_COLORS, NOISE_COLOR } from '@/components/visualizations/ClusterScatterPlot'

interface SelectedResponsesTableProps {
  responses: ClusterCoordinate[]
}

function questionLabel(question?: string): string {
  if (!question) return ''
  if (question.startsWith('What makes')) return 'Q8 — Praise'
  if (question.startsWith('Please provide')) return 'Q9 — Improvement'
  return question.slice(0, 40)
}

export default function SelectedResponsesTable({ responses }: SelectedResponsesTableProps) {
  if (responses.length === 0) return null

  return (
    <div className="border rounded-lg overflow-auto max-h-[400px]">
      <table className="w-full text-sm">
        <thead className="sticky top-0 bg-background border-b">
          <tr>
            <th className="text-left px-3 py-2 w-20">Cluster</th>
            <th className="text-left px-3 py-2 w-24">Level</th>
            <th className="text-left px-3 py-2 w-36">Question</th>
            <th className="text-left px-3 py-2">Response</th>
            <th className="text-left px-3 py-2 w-48">Tags</th>
          </tr>
        </thead>
        <tbody>
          {responses.map((r) => (
            <tr key={r.response_id} className="border-b last:border-b-0 hover:bg-muted/50">
              <td className="px-3 py-2">
                <div className="flex items-center gap-1.5">
                  <div
                    className="w-3 h-3 rounded-full shrink-0"
                    style={{
                      backgroundColor:
                        r.cluster_id === -1
                          ? NOISE_COLOR
                          : CLUSTER_COLORS[r.cluster_id % CLUSTER_COLORS.length],
                    }}
                  />
                  <span>{r.cluster_id === -1 ? 'Noise' : r.cluster_id}</span>
                </div>
              </td>
              <td className="px-3 py-2">
                {r.level && <Badge variant="outline">{r.level}</Badge>}
              </td>
              <td className="px-3 py-2 text-muted-foreground">
                {questionLabel(r.question)}
              </td>
              <td className="px-3 py-2">{r.response_text}</td>
              <td className="px-3 py-2">
                <div className="flex flex-wrap gap-1">
                  {r.tags?.map((tag) => (
                    <Badge key={tag} variant="secondary" className="text-xs">
                      {tag}
                    </Badge>
                  ))}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
