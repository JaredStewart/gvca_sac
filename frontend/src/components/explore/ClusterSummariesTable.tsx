import { useState, useCallback } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import type { ClusterSummary, ClusterMetadata, ClusterCoordinate } from '@/api/client'
import { clusteringApi } from '@/api/client'
import { CLUSTER_COLORS, NOISE_COLOR } from '@/components/visualizations/ClusterScatterPlot'
import { ChevronRight, ChevronDown } from 'lucide-react'
import ClusterResponsesRow from './ClusterResponsesRow'

interface ClusterSummariesTableProps {
  year: string
  summaries: ClusterSummary[]
  metadata: ClusterMetadata[]
  coordinates: ClusterCoordinate[]
  taxonomy: Array<{ name: string; keywords: string[] }>
  filtered?: boolean
}

export default function ClusterSummariesTable({
  year,
  summaries,
  metadata,
  coordinates,
  taxonomy,
  filtered = false,
}: ClusterSummariesTableProps) {
  const queryClient = useQueryClient()
  const [expandedCluster, setExpandedCluster] = useState<number | null>(null)

  // Build metadata lookup
  const metaMap = new Map<number, ClusterMetadata>()
  for (const m of metadata) {
    metaMap.set(m.cluster_id, m)
  }

  // When filters are active, recompute cluster sizes from filtered coordinates
  // and hide clusters with zero matching responses
  const displaySummaries = filtered
    ? (() => {
        const sizeByCid = new Map<number, number>()
        for (const c of coordinates) {
          sizeByCid.set(c.cluster_id, (sizeByCid.get(c.cluster_id) ?? 0) + 1)
        }
        return summaries
          .filter((s) => (sizeByCid.get(s.cluster_id) ?? 0) > 0)
          .map((s) => ({ ...s, size: sizeByCid.get(s.cluster_id) ?? s.size }))
      })()
    : summaries

  // Sort: noise (-1) first, then descending by size
  const sorted = [...displaySummaries].sort((a, b) => {
    if (a.cluster_id === -1) return -1
    if (b.cluster_id === -1) return 1
    return b.size - a.size
  })

  const updateMutation = useMutation({
    mutationFn: ({ clusterId, data }: { clusterId: number; data: { name?: string; description?: string } }) =>
      clusteringApi.updateMetadata(year, clusterId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['cluster-metadata', year] })
    },
  })

  const handleSave = useCallback(
    (clusterId: number, field: 'name' | 'description', value: string) => {
      updateMutation.mutate({ clusterId, data: { [field]: value } })
    },
    [updateMutation],
  )

  const toggleExpand = (clusterId: number) => {
    setExpandedCluster((prev) => (prev === clusterId ? null : clusterId))
  }

  return (
    <div className="border rounded-lg overflow-auto">
      <table className="w-full text-sm">
        <thead className="bg-muted/50 border-b">
          <tr>
            <th className="w-10 px-3 py-2" />
            <th className="text-left px-3 py-2 w-48">Name</th>
            <th className="text-left px-3 py-2">Description</th>
            <th className="text-right px-3 py-2 w-28"># Responses</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((cluster) => {
            const meta = metaMap.get(cluster.cluster_id)
            const defaultName =
              cluster.cluster_id === -1 ? 'Noise' : `Cluster ${cluster.cluster_id}`
            const isExpanded = expandedCluster === cluster.cluster_id
            const clusterColor =
              cluster.cluster_id === -1
                ? NOISE_COLOR
                : CLUSTER_COLORS[cluster.cluster_id % CLUSTER_COLORS.length]

            return (
              <ClusterRow
                key={cluster.cluster_id}
                cluster={cluster}
                meta={meta}
                defaultName={defaultName}
                clusterColor={clusterColor}
                isExpanded={isExpanded}
                onToggleExpand={() => toggleExpand(cluster.cluster_id)}
                onSave={handleSave}
                year={year}
                coordinates={coordinates}
                taxonomy={taxonomy}
              />
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

// Inline-editable cell
function EditableCell({
  value,
  placeholder,
  onSave,
}: {
  value: string
  placeholder: string
  onSave: (value: string) => void
}) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(value)

  const commit = () => {
    setEditing(false)
    if (draft !== value) {
      onSave(draft)
    }
  }

  if (editing) {
    return (
      <input
        className="w-full bg-transparent border-b border-primary outline-none text-sm px-0 py-0.5"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') commit()
          if (e.key === 'Escape') {
            setDraft(value)
            setEditing(false)
          }
        }}
        autoFocus
      />
    )
  }

  return (
    <span
      className="cursor-pointer hover:bg-muted/50 rounded px-1 py-0.5 -mx-1 inline-block min-w-[4rem] text-sm"
      onClick={() => {
        setDraft(value)
        setEditing(true)
      }}
      title="Click to edit"
    >
      {value || <span className="text-muted-foreground italic">{placeholder}</span>}
    </span>
  )
}

// Single cluster row + expandable responses
function ClusterRow({
  cluster,
  meta,
  defaultName,
  clusterColor,
  isExpanded,
  onToggleExpand,
  onSave,
  year,
  coordinates,
  taxonomy,
}: {
  cluster: ClusterSummary
  meta: ClusterMetadata | undefined
  defaultName: string
  clusterColor: string
  isExpanded: boolean
  onToggleExpand: () => void
  onSave: (clusterId: number, field: 'name' | 'description', value: string) => void
  year: string
  coordinates: ClusterCoordinate[]
  taxonomy: Array<{ name: string; keywords: string[] }>
}) {
  const clusterResponses = coordinates.filter((c) => c.cluster_id === cluster.cluster_id)

  return (
    <>
      <tr className="border-b hover:bg-muted/30">
        <td className="px-3 py-2">
          <button onClick={onToggleExpand} className="p-0.5 rounded hover:bg-muted">
            {isExpanded ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <ChevronRight className="h-4 w-4" />
            )}
          </button>
        </td>
        <td className="px-3 py-2">
          <div className="flex items-center gap-2">
            <div
              className="w-3 h-3 rounded-full shrink-0"
              style={{ backgroundColor: clusterColor }}
            />
            <EditableCell
              value={meta?.name || defaultName}
              placeholder={defaultName}
              onSave={(v) => onSave(cluster.cluster_id, 'name', v)}
            />
          </div>
        </td>
        <td className="px-3 py-2">
          <EditableCell
            value={meta?.description || ''}
            placeholder="Add description..."
            onSave={(v) => onSave(cluster.cluster_id, 'description', v)}
          />
        </td>
        <td className="px-3 py-2 text-right font-medium">{cluster.size}</td>
      </tr>
      {isExpanded && (
        <tr>
          <td colSpan={4} className="p-0">
            <ClusterResponsesRow
              year={year}
              responses={clusterResponses}
              taxonomy={taxonomy}
            />
          </td>
        </tr>
      )}
    </>
  )
}
