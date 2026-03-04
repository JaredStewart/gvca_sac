import { useEffect, useState, useRef } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { pipelineApi, dataApi } from '@/api/client'
import { useAppStore } from '@/stores/app'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Loader2, RefreshCw, Database, Cloud } from 'lucide-react'

export default function Header() {
  const queryClient = useQueryClient()
  const { selectedYear, setSelectedYear, years, setYears, pipelineStatus, setPipelineStatus } = useAppStore()
  const [isSyncing, setIsSyncing] = useState(false)
  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const pollTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Cleanup interval/timeout on unmount
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current)
      }
      if (pollTimeoutRef.current) {
        clearTimeout(pollTimeoutRef.current)
      }
    }
  }, [])

  const { data: yearsData } = useQuery({
    queryKey: ['years'],
    queryFn: () => pipelineApi.getYears(),
  })

  // Check if data exists in PocketBase for selected year
  const { data: dataStatus, refetch: refetchDataStatus } = useQuery({
    queryKey: ['data-status', selectedYear],
    queryFn: () => selectedYear ? dataApi.getDataStatus(selectedYear) : null,
    enabled: !!selectedYear,
  })

  // Check pipeline status
  const { data: status, refetch: refetchStatus } = useQuery({
    queryKey: ['pipeline-status', selectedYear],
    queryFn: () => selectedYear ? pipelineApi.getStatus(selectedYear) : null,
    enabled: !!selectedYear,
  })

  // Load data mutation (blocking - loads into memory)
  const loadMutation = useMutation({
    mutationFn: async () => {
      if (!selectedYear) throw new Error('No year selected')
      return pipelineApi.init(selectedYear)
    },
    onSuccess: async (result) => {
      // Update pipeline status
      setPipelineStatus({
        initialized: true,
        loaded: true,
        row_count: result.row_count,
        tagging_complete: false,
        clustering_complete: false,
      })

      // Start async sync to PocketBase
      setIsSyncing(true)
      try {
        await dataApi.syncToPocketbase(selectedYear!, { force: true, runAsync: true })

        // Clear any existing poll interval/timeout
        if (pollIntervalRef.current) {
          clearInterval(pollIntervalRef.current)
        }
        if (pollTimeoutRef.current) {
          clearTimeout(pollTimeoutRef.current)
        }

        // Poll for completion
        pollIntervalRef.current = setInterval(async () => {
          const newStatus = await dataApi.getDataStatus(selectedYear!)
          if (newStatus.has_data) {
            if (pollIntervalRef.current) {
              clearInterval(pollIntervalRef.current)
              pollIntervalRef.current = null
            }
            if (pollTimeoutRef.current) {
              clearTimeout(pollTimeoutRef.current)
              pollTimeoutRef.current = null
            }
            setIsSyncing(false)
            refetchDataStatus()
            queryClient.invalidateQueries({ queryKey: ['survey-responses'] })
            queryClient.invalidateQueries({ queryKey: ['unified-responses'] })
          }
        }, 1000)
        // Timeout after 60 seconds
        pollTimeoutRef.current = setTimeout(() => {
          if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current)
            pollIntervalRef.current = null
          }
          setIsSyncing(false)
        }, 60000)
      } catch {
        setIsSyncing(false)
      }

      refetchStatus()
    },
  })

  useEffect(() => {
    if (yearsData?.years) {
      setYears(yearsData.years)
      if (!selectedYear && yearsData.years.length > 0) {
        setSelectedYear(yearsData.years[0])
      }
    }
  }, [yearsData, selectedYear, setSelectedYear, setYears])

  // Update global pipeline status from query
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

  const handleLoadData = () => {
    loadMutation.mutate()
  }

  const isLoading = loadMutation.isPending
  const hasDataInDb = dataStatus?.has_data ?? false
  const hasDataInMemory = pipelineStatus?.loaded ?? false
  const needsLoad = !hasDataInDb && !hasDataInMemory

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="flex h-14 items-center px-6">
        <div className="flex items-center space-x-4">
          <h1 className="text-lg font-semibold">GVCA SAC Survey Analysis</h1>
        </div>

        <div className="ml-auto flex items-center space-x-4">
          {/* Status badges */}
          {selectedYear && (
            <div className="flex items-center space-x-2">
              {(hasDataInDb || hasDataInMemory) && (
                <Badge variant="secondary" className="gap-1">
                  <Database className="h-3 w-3" />
                  {dataStatus?.survey_response_count ?? pipelineStatus?.row_count ?? 0} responses
                </Badge>
              )}
              {isSyncing && (
                <Badge variant="outline" className="gap-1 animate-pulse">
                  <Cloud className="h-3 w-3" />
                  Syncing...
                </Badge>
              )}
              {pipelineStatus?.tagging_complete && (
                <Badge variant="default">Tagged</Badge>
              )}
              {pipelineStatus?.clustering_complete && (
                <Badge variant="default">Clustered</Badge>
              )}
            </div>
          )}

          {/* Load/Refresh button */}
          {selectedYear && (
            <Button
              variant={needsLoad ? 'default' : 'outline'}
              size="sm"
              onClick={handleLoadData}
              disabled={isLoading || isSyncing}
            >
              {isLoading ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Loading...
                </>
              ) : needsLoad ? (
                <>
                  <Database className="h-4 w-4 mr-2" />
                  Load Data
                </>
              ) : (
                <>
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Refresh
                </>
              )}
            </Button>
          )}

          {/* Year selector */}
          <Select value={selectedYear || ''} onValueChange={setSelectedYear}>
            <SelectTrigger className="w-[120px]">
              <SelectValue placeholder="Select year" />
            </SelectTrigger>
            <SelectContent>
              {years.map((year) => (
                <SelectItem key={year} value={year}>
                  {year}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>
    </header>
  )
}
