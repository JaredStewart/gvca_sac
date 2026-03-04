import type { ClusterCoordinate } from '@/api/client'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { X } from 'lucide-react'

interface ResponseDetailPanelProps {
  response: ClusterCoordinate | null
  onClose: () => void
}

const CLUSTER_COLORS = [
  '#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#00C49F',
  '#FFBB28', '#FF8042', '#0088FE', '#00C49F', '#FFBB28',
  '#9C27B0', '#E91E63', '#3F51B5', '#009688', '#795548',
]

export default function ResponseDetailPanel({ response, onClose }: ResponseDetailPanelProps) {
  if (!response) return null

  const clusterColor = response.cluster_id === -1
    ? '#ccc'
    : CLUSTER_COLORS[response.cluster_id % CLUSTER_COLORS.length]

  return (
    <Card className="w-80">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div
              className="w-4 h-4 rounded-full"
              style={{ backgroundColor: clusterColor }}
            />
            <CardTitle className="text-sm">
              {response.cluster_id === -1 ? 'Noise' : `Cluster ${response.cluster_id}`}
            </CardTitle>
          </div>
          <button
            onClick={onClose}
            className="p-1 hover:bg-muted rounded"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <CardDescription className="text-xs">
          Response ID: {response.response_id}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {response.level && (
          <div>
            <div className="text-xs font-medium text-muted-foreground mb-1">Level</div>
            <Badge variant="outline">{response.level}</Badge>
          </div>
        )}

        {response.question && (
          <div>
            <div className="text-xs font-medium text-muted-foreground mb-1">Question</div>
            <p className="text-xs">{response.question}</p>
          </div>
        )}

        {response.response_text && (
          <div>
            <div className="text-xs font-medium text-muted-foreground mb-1">Response</div>
            <p className="text-sm border-l-2 pl-2 italic">{response.response_text}</p>
          </div>
        )}

        {response.tags && response.tags.length > 0 && (
          <div>
            <div className="text-xs font-medium text-muted-foreground mb-1">Tags</div>
            <div className="flex flex-wrap gap-1">
              {response.tags.map(tag => (
                <Badge key={tag} variant="secondary" className="text-xs">
                  {tag}
                </Badge>
              ))}
            </div>
          </div>
        )}

        <div className="text-xs text-muted-foreground pt-2 border-t">
          <div>UMAP: ({response.x.toFixed(2)}, {response.y.toFixed(2)})</div>
        </div>
      </CardContent>
    </Card>
  )
}
