import { useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { X, Loader2, CheckCircle, XCircle } from 'lucide-react'
import { useAppStore } from '@/stores/app'
import { jobsApi, subscribeToJob, type Job } from '@/api/client'

interface JobProgressProps {
  jobId: string
  onComplete?: () => void
}

export default function JobProgress({ jobId, onComplete }: JobProgressProps) {
  const { activeJobs, updateJob, removeJob } = useAppStore()
  const job = activeJobs.find((j) => j.id === jobId)

  useEffect(() => {
    // Fetch initial status
    jobsApi.getStatus(jobId).then((data) => {
      updateJob(jobId, data)
    })

    // Subscribe to WebSocket updates
    const unsubscribe = subscribeToJob(jobId, (data) => {
      updateJob(jobId, data)
      if (data.status === 'completed' || data.status === 'failed' || data.status === 'cancelled') {
        onComplete?.()
      }
    })

    return () => {
      unsubscribe()
    }
  }, [jobId, updateJob, onComplete])

  if (!job) {
    return null
  }

  const getStatusIcon = () => {
    switch (job.status) {
      case 'running':
        return <Loader2 className="h-4 w-4 animate-spin" />
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-600" />
      case 'failed':
        return <XCircle className="h-4 w-4 text-red-600" />
      default:
        return null
    }
  }

  const handleCancel = async () => {
    await jobsApi.cancel(jobId)
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          {getStatusIcon()}
          {job.job_type.replace('_', ' ').toUpperCase()}
        </CardTitle>
        <div className="flex items-center gap-2">
          <Badge variant={job.status === 'completed' ? 'default' : 'secondary'}>
            {job.status}
          </Badge>
          {job.status === 'running' && (
            <Button variant="ghost" size="icon" onClick={handleCancel}>
              <X className="h-4 w-4" />
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <Progress value={job.progress} className="h-2" />
        <div className="mt-2 flex justify-between text-xs text-muted-foreground">
          <span>
            {job.processed_items} / {job.total_items} items
          </span>
          <span>{Math.round(job.progress)}%</span>
        </div>
        {job.error_message && (
          <p className="mt-2 text-xs text-red-600">{job.error_message}</p>
        )}
      </CardContent>
    </Card>
  )
}

export function ActiveJobs() {
  const { activeJobs } = useAppStore()

  if (activeJobs.length === 0) {
    return null
  }

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">Active Jobs</h3>
      {activeJobs.map((job) => (
        <JobProgress key={job.id} jobId={job.id} />
      ))}
    </div>
  )
}
