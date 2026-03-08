import { create } from 'zustand'
import type { Job } from '@/api/client'

interface AppState {
  // Selected year
  selectedYear: string | null
  setSelectedYear: (year: string | null) => void

  // Available years
  years: string[]
  setYears: (years: string[]) => void

  // Pipeline status
  pipelineStatus: {
    initialized: boolean
    loaded: boolean
    row_count: number
    tagging_complete: boolean
    clustering_complete: boolean
  } | null
  setPipelineStatus: (status: AppState['pipelineStatus']) => void

  // Family weighting toggle
  weightByParents: boolean
  toggleWeightByParents: () => void

  // Active jobs
  activeJobs: Job[]
  addJob: (job: Job) => void
  updateJob: (jobId: string, updates: Partial<Job>) => void
  removeJob: (jobId: string) => void
}

export const useAppStore = create<AppState>((set) => ({
  selectedYear: null,
  setSelectedYear: (year) => set({ selectedYear: year }),

  years: [],
  setYears: (years) => set({ years }),

  pipelineStatus: null,
  setPipelineStatus: (status) => set({ pipelineStatus: status }),

  weightByParents: true,
  toggleWeightByParents: () => set((state) => ({ weightByParents: !state.weightByParents })),

  activeJobs: [],
  addJob: (job) =>
    set((state) => ({
      activeJobs: [...state.activeJobs.filter((j) => j.id !== job.id), job],
    })),
  updateJob: (jobId, updates) =>
    set((state) => ({
      activeJobs: state.activeJobs.map((j) =>
        j.id === jobId ? { ...j, ...updates } : j
      ),
    })),
  removeJob: (jobId) =>
    set((state) => ({
      activeJobs: state.activeJobs.filter((j) => j.id !== jobId),
    })),
}))
