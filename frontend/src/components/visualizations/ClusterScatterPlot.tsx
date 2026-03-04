import { useRef, useEffect, useState, useCallback } from 'react'
import * as d3 from 'd3'
import type { ClusterCoordinate } from '@/api/client'

interface ClusterScatterPlotProps {
  data: ClusterCoordinate[]
  width?: number
  height?: number
  onPointClick?: (point: ClusterCoordinate) => void
  onSelectionChange?: (points: ClusterCoordinate[]) => void
  filterLevel?: string
  filterTag?: string
  filterQuestion?: string
  hideLegend?: boolean
}

export const CLUSTER_COLORS = [
  '#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#00C49F',
  '#FFBB28', '#FF8042', '#0088FE', '#00C49F', '#FFBB28',
  '#9C27B0', '#E91E63', '#3F51B5', '#009688', '#795548',
]

export const NOISE_COLOR = '#ccc'

export default function ClusterScatterPlot({
  data,
  width = 800,
  height = 500,
  onPointClick,
  onSelectionChange,
  filterLevel,
  filterTag,
  filterQuestion,
  hideLegend,
}: ClusterScatterPlotProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [selectedPoints, setSelectedPoints] = useState<Set<string>>(new Set())
  const [isLassoMode, setIsLassoMode] = useState(false)
  const transformRef = useRef<d3.ZoomTransform>(d3.zoomIdentity)

  // Filter data based on level, tag, and question
  const filteredData = !Array.isArray(data) ? [] : data.filter(point => {
    if (filterLevel && point.level && point.level !== filterLevel) {
      return false
    }
    if (filterTag && point.tags && !point.tags.includes(filterTag)) {
      return false
    }
    if (filterQuestion && point.question && point.question !== filterQuestion) {
      return false
    }
    return true
  })

  const getPointColor = useCallback((point: ClusterCoordinate) => {
    if (point.cluster_id === -1) return NOISE_COLOR
    return CLUSTER_COLORS[point.cluster_id % CLUSTER_COLORS.length]
  }, [])

  // Draw points on canvas for performance
  const drawCanvas = useCallback((transform: d3.ZoomTransform) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const margin = { top: 20, right: 20, bottom: 40, left: 50 }
    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    // Create scales
    const xExtent = d3.extent(data, d => d.x) as [number, number]
    const yExtent = d3.extent(data, d => d.y) as [number, number]

    const xScale = d3.scaleLinear()
      .domain([xExtent[0] - 0.5, xExtent[1] + 0.5])
      .range([margin.left, margin.left + innerWidth])

    const yScale = d3.scaleLinear()
      .domain([yExtent[0] - 0.5, yExtent[1] + 0.5])
      .range([margin.top + innerHeight, margin.top])

    // Apply transform to scales
    const xScaleZoomed = transform.rescaleX(xScale)
    const yScaleZoomed = transform.rescaleY(yScale)

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    // Draw points
    filteredData.forEach(point => {
      const x = xScaleZoomed(point.x)
      const y = yScaleZoomed(point.y)

      // Skip points outside visible area
      if (x < margin.left || x > width - margin.right ||
          y < margin.top || y > height - margin.bottom) {
        return
      }

      ctx.beginPath()
      ctx.arc(x, y, selectedPoints.has(point.response_id) ? 6 : 4, 0, Math.PI * 2)
      ctx.fillStyle = getPointColor(point)
      ctx.globalAlpha = selectedPoints.has(point.response_id) ? 1 : 0.7
      ctx.fill()

      if (selectedPoints.has(point.response_id)) {
        ctx.strokeStyle = '#000'
        ctx.lineWidth = 2
        ctx.stroke()
      }
    })

    ctx.globalAlpha = 1
  }, [data, filteredData, width, height, selectedPoints, getPointColor])

  useEffect(() => {
    const svg = d3.select(svgRef.current)
    const canvas = canvasRef.current
    if (!svg.node() || !canvas) return

    const margin = { top: 20, right: 20, bottom: 40, left: 50 }
    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    // Clear previous content
    svg.selectAll('*').remove()

    // Create scales
    const xExtent = d3.extent(data, d => d.x) as [number, number]
    const yExtent = d3.extent(data, d => d.y) as [number, number]

    const xScale = d3.scaleLinear()
      .domain([xExtent[0] - 0.5, xExtent[1] + 0.5])
      .range([margin.left, margin.left + innerWidth])

    const yScale = d3.scaleLinear()
      .domain([yExtent[0] - 0.5, yExtent[1] + 0.5])
      .range([margin.top + innerHeight, margin.top])

    // Create axes
    const xAxis = d3.axisBottom(xScale).ticks(5)
    const yAxis = d3.axisLeft(yScale).ticks(5)

    // Add axes to SVG
    const gX = svg.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${margin.top + innerHeight})`)
      .call(xAxis)

    const gY = svg.append('g')
      .attr('class', 'y-axis')
      .attr('transform', `translate(${margin.left},0)`)
      .call(yAxis)

    // Add axis labels
    svg.append('text')
      .attr('x', margin.left + innerWidth / 2)
      .attr('y', height - 5)
      .attr('text-anchor', 'middle')
      .attr('class', 'text-xs fill-muted-foreground')
      .text('UMAP 1')

    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -(margin.top + innerHeight / 2))
      .attr('y', 15)
      .attr('text-anchor', 'middle')
      .attr('class', 'text-xs fill-muted-foreground')
      .text('UMAP 2')

    // Create a clip path
    svg.append('defs')
      .append('clipPath')
      .attr('id', 'clip')
      .append('rect')
      .attr('x', margin.left)
      .attr('y', margin.top)
      .attr('width', innerWidth)
      .attr('height', innerHeight)

    // Lasso path
    const lassoPath = svg.append('path')
      .attr('class', 'lasso')
      .style('fill', 'rgba(100, 100, 255, 0.2)')
      .style('stroke', '#6666ff')
      .style('stroke-width', 2)
      .style('stroke-dasharray', '4')
      .attr('clip-path', 'url(#clip)')

    let lassoPoints: [number, number][] = []

    // Zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.5, 10])
      .translateExtent([[0, 0], [width, height]])
      .on('zoom', (event) => {
        transformRef.current = event.transform

        // Update axes
        const newXScale = event.transform.rescaleX(xScale)
        const newYScale = event.transform.rescaleY(yScale)
        gX.call(d3.axisBottom(newXScale).ticks(5))
        gY.call(d3.axisLeft(newYScale).ticks(5))

        // Redraw canvas
        drawCanvas(event.transform)
      })

    if (!isLassoMode) {
      svg.call(zoom)
    }

    // Lasso drag behavior
    const lassoDrag = d3.drag<SVGSVGElement, unknown>()
      .on('start', (event) => {
        lassoPoints = [[event.x, event.y]]
        lassoPath.attr('d', '')
      })
      .on('drag', (event) => {
        lassoPoints.push([event.x, event.y])
        const lineGenerator = d3.line()
        lassoPath.attr('d', lineGenerator(lassoPoints) + 'Z')
      })
      .on('end', () => {
        if (lassoPoints.length < 3) {
          lassoPath.attr('d', '')
          return
        }

        // Create polygon from lasso points
        const polygon = lassoPoints

        // Find points inside polygon
        const transform = transformRef.current
        const xScaleZoomed = transform.rescaleX(xScale)
        const yScaleZoomed = transform.rescaleY(yScale)

        const selectedIds = new Set<string>()
        filteredData.forEach(point => {
          const x = xScaleZoomed(point.x)
          const y = yScaleZoomed(point.y)
          if (pointInPolygon([x, y], polygon)) {
            selectedIds.add(point.response_id)
          }
        })

        setSelectedPoints(selectedIds)
        if (onSelectionChange) {
          const selected = filteredData.filter(p => selectedIds.has(p.response_id))
          onSelectionChange(selected)
        }

        lassoPath.attr('d', '')
      })

    if (isLassoMode) {
      svg.call(lassoDrag as unknown as d3.DragBehavior<SVGSVGElement, unknown, unknown>)
    }

    // Click handler for individual points
    svg.on('click', (event) => {
      if (isLassoMode) return

      const transform = transformRef.current
      const xScaleZoomed = transform.rescaleX(xScale)
      const yScaleZoomed = transform.rescaleY(yScale)

      const [mouseX, mouseY] = d3.pointer(event)

      // Find closest point
      let closestPoint: ClusterCoordinate | null = null
      let closestDist = Infinity

      filteredData.forEach(point => {
        const x = xScaleZoomed(point.x)
        const y = yScaleZoomed(point.y)
        const dist = Math.sqrt((x - mouseX) ** 2 + (y - mouseY) ** 2)
        if (dist < closestDist && dist < 10) {
          closestDist = dist
          closestPoint = point
        }
      })

      if (closestPoint && onPointClick) {
        onPointClick(closestPoint)
      }
    })

    // Initial draw
    drawCanvas(transformRef.current)

    // Cleanup
    return () => {
      svg.on('.zoom', null)
      svg.on('click', null)
    }
  }, [data, filteredData, width, height, isLassoMode, drawCanvas, onPointClick, onSelectionChange])

  // Redraw when selection changes
  useEffect(() => {
    drawCanvas(transformRef.current)
  }, [selectedPoints, drawCanvas])

  // Clear selection
  const clearSelection = () => {
    setSelectedPoints(new Set())
    if (onSelectionChange) {
      onSelectionChange([])
    }
  }

  return (
    <div className="relative">
      <div className="absolute top-2 right-2 z-10 flex gap-2">
        <button
          className={`px-3 py-1 text-sm rounded ${isLassoMode ? 'bg-primary text-primary-foreground' : 'bg-secondary'}`}
          onClick={() => setIsLassoMode(!isLassoMode)}
        >
          {isLassoMode ? 'Pan Mode' : 'Lasso Select'}
        </button>
        {selectedPoints.size > 0 && (
          <button
            className="px-3 py-1 text-sm rounded bg-secondary"
            onClick={clearSelection}
          >
            Clear ({selectedPoints.size})
          </button>
        )}
      </div>
      <div className="relative">
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}
        />
        <svg
          ref={svgRef}
          width={width}
          height={height}
          style={{ cursor: isLassoMode ? 'crosshair' : 'grab' }}
        />
      </div>
      {!hideLegend && (
        <div className="mt-2 flex flex-wrap gap-2">
          {Array.from(new Set(filteredData.map(d => d.cluster_id)))
            .filter(id => id !== -1)
            .sort((a, b) => a - b)
            .map(clusterId => (
              <div key={clusterId} className="flex items-center gap-1 text-xs">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: CLUSTER_COLORS[clusterId % CLUSTER_COLORS.length] }}
                />
                <span>Cluster {clusterId}</span>
              </div>
            ))}
          {filteredData.some(d => d.cluster_id === -1) && (
            <div className="flex items-center gap-1 text-xs">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: NOISE_COLOR }} />
              <span>Noise</span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// Point-in-polygon test using ray casting
function pointInPolygon(point: [number, number], polygon: [number, number][]): boolean {
  const [x, y] = point
  let inside = false

  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i]
    const [xj, yj] = polygon[j]

    if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
      inside = !inside
    }
  }

  return inside
}
