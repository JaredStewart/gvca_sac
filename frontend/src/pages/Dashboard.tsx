import { useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { pipelineApi, dataApi, taggingApi } from '@/api/client'
import { useAppStore } from '@/stores/app'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import StatsCard from '@/components/dashboard/StatsCard'
import { ActiveJobs } from '@/components/jobs/JobProgress'
import { Users, FileText, Tags, Layers, RefreshCw } from 'lucide-react'

export default function Dashboard() {
  const { selectedYear, setPipelineStatus } = useAppStore()

  // Fetch pipeline status
  const { data: status, refetch: refetchStatus } = useQuery({
    queryKey: ['pipeline-status', selectedYear],
    queryFn: () => selectedYear ? pipelineApi.getStatus(selectedYear) : null,
    enabled: !!selectedYear,
  })

  // Fetch statistics when loaded
  const { data: statistics } = useQuery({
    queryKey: ['statistics', selectedYear],
    queryFn: () => selectedYear ? dataApi.getStatistics(selectedYear) : null,
    enabled: !!selectedYear && !!status?.loaded,
  })

  // Fetch tag distribution when tagging complete
  const { data: tagDistribution } = useQuery({
    queryKey: ['tag-distribution', selectedYear],
    queryFn: () => selectedYear ? taggingApi.getDistribution(selectedYear) : null,
    enabled: !!selectedYear && !!status?.tagging_complete,
  })

  // Fetch cross-year status
  const { data: allStatus } = useQuery({
    queryKey: ['pipeline-status-all'],
    queryFn: () => pipelineApi.getAllStatus(),
  })

  // Update global pipeline status
  useEffect(() => {
    if (status) {
      setPipelineStatus({
        initialized: status.initialized,
        loaded: status.loaded,
        row_count: status.row_count,
        tagging_complete: status.tagging_complete,
        clustering_complete: status.clustering_complete,
      })
    }
  }, [status, setPipelineStatus])

  if (!selectedYear) {
    return (
      <div className="flex items-center justify-center h-[50vh]">
        <Card className="w-96">
          <CardHeader>
            <CardTitle>No Year Selected</CardTitle>
            <CardDescription>
              Select a survey year from the dropdown to get started.
            </CardDescription>
          </CardHeader>
        </Card>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold tracking-tight">{selectedYear} Dashboard</h2>
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => refetchStatus()}
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Active Jobs */}
      <ActiveJobs />

      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <StatsCard
          title="Total Responses"
          value={statistics?.total_responses ?? status?.row_count ?? 0}
          icon={<Users className="h-4 w-4" />}
          description="survey responses"
        />
        <StatsCard
          title="Overall Average"
          value={statistics?.weighted_averages?.['Overall Average']?.toFixed(2) ?? '-'}
          icon={<FileText className="h-4 w-4" />}
          description="satisfaction score"
        />
        <StatsCard
          title="Unique Tags"
          value={tagDistribution?.unique_tags ?? '-'}
          icon={<Tags className="h-4 w-4" />}
          description={status?.tagging_complete ? 'tags assigned' : 'not tagged yet'}
        />
        <StatsCard
          title="Status"
          value={status?.loaded ? 'Ready' : 'Not Loaded'}
          icon={<Layers className="h-4 w-4" />}
          description={status?.last_updated ? `Updated ${new Date(status.last_updated).toLocaleDateString()}` : ''}
        />
      </div>

      {/* Level Breakdown */}
      {statistics?.level_counts && Object.keys(statistics.level_counts).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Responses by Level</CardTitle>
            <CardDescription>Distribution across school levels</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-3">
              {Object.entries(statistics.level_counts).map(([level, count]) => (
                <div key={level} className="flex items-center justify-between p-4 border rounded-lg">
                  <span className="font-medium">{level}</span>
                  <span className="text-2xl font-bold">{count as number}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Weighted Averages */}
      {statistics?.weighted_averages && (
        <Card>
          <CardHeader>
            <CardTitle>Weighted Averages</CardTitle>
            <CardDescription>Average satisfaction by level (1-4 scale)</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-4">
              {Object.entries(statistics.weighted_averages).map(([level, avg]) => (
                <div key={level} className="text-center p-4 border rounded-lg">
                  <div className="text-sm text-muted-foreground">{level.replace(' Average', '')}</div>
                  <div className="text-3xl font-bold mt-1">
                    {typeof avg === 'number' ? avg.toFixed(2) : '-'}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Year Status Overview */}
      {allStatus?.years && allStatus.years.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Year Status</CardTitle>
            <CardDescription>Overview of data processing across all years</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-2 pr-4 font-medium">Year</th>
                    <th className="text-left py-2 pr-4 font-medium">Responses</th>
                    <th className="text-left py-2 pr-4 font-medium">Tagging</th>
                    <th className="text-left py-2 pr-4 font-medium">Clustering</th>
                  </tr>
                </thead>
                <tbody>
                  {allStatus.years.map((ys) => (
                    <tr key={ys.year} className="border-b last:border-0">
                      <td className="py-2 pr-4">
                        <span className="font-medium">{ys.year}</span>
                        {ys.year === selectedYear && (
                          <Badge variant="outline" className="ml-2 text-xs">selected</Badge>
                        )}
                      </td>
                      <td className="py-2 pr-4">
                        {ys.loaded ? (
                          <Badge variant="default" className="bg-green-600">{ys.row_count}</Badge>
                        ) : (
                          <Badge variant="secondary">not loaded</Badge>
                        )}
                      </td>
                      <td className="py-2 pr-4">
                        {ys.tagging_count > 0 ? (
                          <Badge variant="default" className={ys.tagging_complete ? 'bg-green-600' : ''}>
                            {ys.tagging_count}
                          </Badge>
                        ) : (
                          <Badge variant="secondary">none</Badge>
                        )}
                      </td>
                      <td className="py-2 pr-4">
                        {ys.clustering_count > 0 ? (
                          <Badge variant="default" className={ys.clustering_complete ? 'bg-green-600' : ''}>
                            {ys.clustering_count}
                          </Badge>
                        ) : (
                          <Badge variant="secondary">none</Badge>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
