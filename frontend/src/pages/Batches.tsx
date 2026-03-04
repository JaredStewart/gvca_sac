import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { batchJobsApi } from '@/api/client'
import type { BatchJob } from '@/api/client'
import { useAppStore } from '@/stores/app'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { ListChecks, RefreshCw, Trash2, XCircle, RotateCcw } from 'lucide-react'
import { useState } from 'react'

const TERMINAL_STATES = new Set(['completed', 'failed', 'expired', 'cancelled', 'superseded'])
const ACTIVE_STATES = new Set(['queued', 'submitting', 'validating', 'in_progress', 'finalizing'])

function statusBadge(status: string) {
  switch (status) {
    case 'completed':
      return <Badge variant="default" className="bg-green-600">{status}</Badge>
    case 'in_progress':
    case 'validating':
    case 'submitting':
      return <Badge variant="default" className="bg-yellow-500 text-black">{status}</Badge>
    case 'queued':
      return <Badge variant="secondary">{status}</Badge>
    case 'failed':
    case 'expired':
    case 'cancelled':
      return <Badge variant="destructive">{status}</Badge>
    case 'superseded':
      return <Badge variant="outline">{status}</Badge>
    default:
      return <Badge variant="secondary">{status}</Badge>
  }
}

function formatDate(dateStr: string | null | undefined) {
  if (!dateStr) return '-'
  try {
    return new Date(dateStr).toLocaleString()
  } catch {
    return dateStr
  }
}

function ProgressBar({ processed, total }: { processed: number; total: number }) {
  const pct = total > 0 ? Math.round((processed / total) * 100) : 0
  return (
    <div className="flex items-center gap-2">
      <div className="w-24 h-2 bg-muted rounded-full overflow-hidden">
        <div
          className="h-full bg-primary rounded-full transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs text-muted-foreground">{pct}%</span>
    </div>
  )
}

export default function Batches() {
  const { selectedYear } = useAppStore()
  const [expandedRow, setExpandedRow] = useState<string | null>(null)
  const queryClient = useQueryClient()

  const { data, isLoading } = useQuery({
    queryKey: ['batch-jobs', selectedYear],
    queryFn: () => batchJobsApi.list(selectedYear || undefined),
    refetchInterval: (query) => {
      const items = query.state.data?.items ?? []
      const hasActive = items.some(
        (j: BatchJob) => !TERMINAL_STATES.has(j.status)
      )
      return hasActive ? 10_000 : false
    },
  })

  const invalidate = () => queryClient.invalidateQueries({ queryKey: ['batch-jobs'] })

  const deleteMutation = useMutation({
    mutationFn: (id: string) => batchJobsApi.delete(id),
    onSuccess: invalidate,
  })

  const cancelMutation = useMutation({
    mutationFn: (id: string) => batchJobsApi.cancel(id),
    onSuccess: invalidate,
  })

  const retryMutation = useMutation({
    mutationFn: (id: string) => batchJobsApi.retry(id),
    onSuccess: invalidate,
  })

  const pollMutation = useMutation({
    mutationFn: (id: string) => batchJobsApi.poll(id),
    onSuccess: invalidate,
  })

  const clearAllMutation = useMutation({
    mutationFn: () => batchJobsApi.clearAll(selectedYear || undefined),
    onSuccess: invalidate,
  })

  const handleClearAll = () => {
    if (!window.confirm('Clear all batch jobs? Active batches will be cancelled.')) return
    clearAllMutation.mutate()
  }

  const pollAllActive = async () => {
    const activeJobs = jobs.filter((j) => ACTIVE_STATES.has(j.status) && j.openai_batch_id)
    await Promise.allSettled(activeJobs.map((j) => batchJobsApi.poll(j.id)))
    invalidate()
  }

  const jobs = data?.items ?? []
  const hasActiveJobs = jobs.some((j) => ACTIVE_STATES.has(j.status))

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold tracking-tight">Batch Jobs</h2>
        <div className="flex items-center gap-2">
          {hasActiveJobs && (
            <Button
              variant="outline"
              size="sm"
              onClick={pollAllActive}
            >
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh All
            </Button>
          )}
          {jobs.length > 0 && (
            <Button
              variant="destructive"
              size="sm"
              onClick={handleClearAll}
              disabled={clearAllMutation.isPending}
            >
              <Trash2 className="h-4 w-4 mr-2" />
              Clear All
            </Button>
          )}
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <ListChecks className="h-5 w-5" />
            Batch History
          </CardTitle>
          <CardDescription>
            {selectedYear
              ? `Batch jobs for ${selectedYear}`
              : 'All batch jobs'}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <p className="text-sm text-muted-foreground">Loading...</p>
          ) : jobs.length === 0 ? (
            <p className="text-sm text-muted-foreground">No batch jobs found.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-2 px-2 font-medium">Type</th>
                    <th className="text-left py-2 px-2 font-medium">Year</th>
                    <th className="text-left py-2 px-2 font-medium">Status</th>
                    <th className="text-left py-2 px-2 font-medium">Progress</th>
                    <th className="text-left py-2 px-2 font-medium">Items</th>
                    <th className="text-left py-2 px-2 font-medium">Model</th>
                    <th className="text-left py-2 px-2 font-medium">OpenAI Batch ID</th>
                    <th className="text-left py-2 px-2 font-medium">Created</th>
                    <th className="text-left py-2 px-2 font-medium">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {jobs.map((job: BatchJob) => (
                    <>
                      <tr
                        key={job.id}
                        className="border-b hover:bg-muted/50 cursor-pointer"
                        onClick={() =>
                          setExpandedRow(expandedRow === job.id ? null : job.id)
                        }
                      >
                        <td className="py-2 px-2">{job.job_type}</td>
                        <td className="py-2 px-2">{job.year}</td>
                        <td className="py-2 px-2">{statusBadge(job.status)}</td>
                        <td className="py-2 px-2">
                          <ProgressBar
                            processed={job.processed_items}
                            total={job.total_items}
                          />
                        </td>
                        <td className="py-2 px-2 whitespace-nowrap">
                          {job.processed_items}/{job.total_items}
                          {job.failed_items > 0 && (
                            <span className="text-destructive ml-1">
                              ({job.failed_items} failed)
                            </span>
                          )}
                        </td>
                        <td className="py-2 px-2 font-mono text-xs">
                          {job.model_used || '-'}
                        </td>
                        <td className="py-2 px-2 font-mono text-xs">
                          {job.openai_batch_id
                            ? job.openai_batch_id.slice(0, 20) + '...'
                            : '-'}
                        </td>
                        <td className="py-2 px-2 whitespace-nowrap">
                          {formatDate(job.started_at || job.created)}
                        </td>
                        <td className="py-2 px-2" onClick={(e) => e.stopPropagation()}>
                          <div className="flex items-center gap-1">
                            {ACTIVE_STATES.has(job.status) && job.openai_batch_id && (
                              <>
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  className="h-7 w-7"
                                  title="Refresh status"
                                  onClick={() => pollMutation.mutate(job.id)}
                                  disabled={pollMutation.isPending}
                                >
                                  <RefreshCw className="h-3.5 w-3.5" />
                                </Button>
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  className="h-7 w-7 text-destructive"
                                  title="Cancel batch"
                                  onClick={() => cancelMutation.mutate(job.id)}
                                  disabled={cancelMutation.isPending}
                                >
                                  <XCircle className="h-3.5 w-3.5" />
                                </Button>
                              </>
                            )}
                            {job.status === 'failed' && (
                              <Button
                                variant="ghost"
                                size="icon"
                                className="h-7 w-7"
                                title="Retry (split & resubmit)"
                                onClick={() => retryMutation.mutate(job.id)}
                                disabled={retryMutation.isPending}
                              >
                                <RotateCcw className="h-3.5 w-3.5" />
                              </Button>
                            )}
                            {TERMINAL_STATES.has(job.status) && (
                              <Button
                                variant="ghost"
                                size="icon"
                                className="h-7 w-7 text-muted-foreground hover:text-destructive"
                                title="Delete record"
                                onClick={() => deleteMutation.mutate(job.id)}
                                disabled={deleteMutation.isPending}
                              >
                                <Trash2 className="h-3.5 w-3.5" />
                              </Button>
                            )}
                          </div>
                        </td>
                      </tr>
                      {expandedRow === job.id && (
                        <tr key={`${job.id}-detail`} className="bg-muted/30">
                          <td colSpan={9} className="py-3 px-4">
                            <div className="space-y-2 text-sm">
                              {job.error_message && (
                                <div>
                                  <span className="font-medium text-destructive">Error: </span>
                                  <span>{job.error_message}</span>
                                </div>
                              )}
                              <div className="grid grid-cols-2 gap-x-8 gap-y-1">
                                <div>
                                  <span className="text-muted-foreground">Record ID: </span>
                                  <span className="font-mono">{job.id}</span>
                                </div>
                                <div>
                                  <span className="text-muted-foreground">OpenAI Batch ID: </span>
                                  <span className="font-mono">{job.openai_batch_id || '-'}</span>
                                </div>
                                <div>
                                  <span className="text-muted-foreground">Input File: </span>
                                  <span className="font-mono">{job.input_file_id || '-'}</span>
                                </div>
                                <div>
                                  <span className="text-muted-foreground">Output File: </span>
                                  <span className="font-mono">{job.output_file_id || '-'}</span>
                                </div>
                                {job.batch_group_id && (
                                  <div>
                                    <span className="text-muted-foreground">Batch Group: </span>
                                    <span className="font-mono">{job.batch_group_id}</span>
                                  </div>
                                )}
                                {job.estimated_tokens != null && (
                                  <div>
                                    <span className="text-muted-foreground">Est. Tokens: </span>
                                    <span>{job.estimated_tokens.toLocaleString()}</span>
                                  </div>
                                )}
                                <div>
                                  <span className="text-muted-foreground">Completed: </span>
                                  <span>{formatDate(job.completed_at)}</span>
                                </div>
                              </div>
                            </div>
                          </td>
                        </tr>
                      )}
                    </>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
