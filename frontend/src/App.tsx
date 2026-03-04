import { Routes, Route } from 'react-router-dom'
import { ErrorBoundary } from './components/ErrorBoundary'
import Layout from './components/layout/Layout'
import Dashboard from './pages/Dashboard'
import Tagging from './pages/Tagging'
import Visualizations from './pages/Visualizations'
import Explore from './pages/Explore'
import DataTable from './pages/DataTable'
import Batches from './pages/Batches'

function App() {
  return (
    <ErrorBoundary>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="data" element={<DataTable />} />
          <Route path="tagging" element={<Tagging />} />
          <Route path="explore" element={<Explore />} />
          <Route path="visualizations" element={<Visualizations />} />
          <Route path="batches" element={<Batches />} />
        </Route>
      </Routes>
    </ErrorBoundary>
  )
}

export default App
