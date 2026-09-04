import { Navigate, type RouteObject } from 'react-router-dom'
import { WorkbenchLayout } from './layouts/WorkbenchLayout'
import { DatasetDetailPage } from './pages/DatasetDetailPage'
import { DatasetsPage } from './pages/DatasetsPage'
import { InferencePage } from './pages/InferencePage'
import { ModelsPage } from './pages/ModelsPage'
import { NotFoundPage } from './pages/NotFoundPage'
import { OverviewPage } from './pages/OverviewPage'
import { RunDetailPage } from './pages/RunDetailPage'
import { RunsPage } from './pages/RunsPage'
import { TrainingPage } from './pages/TrainingPage'

export const routes: RouteObject[] = [
  {
    path: '/',
    element: <WorkbenchLayout />,
    children: [
      { index: true, element: <Navigate to="/overview" replace /> },
      { path: 'overview', element: <OverviewPage /> },
      { path: 'datasets', element: <DatasetsPage /> },
      { path: 'datasets/:datasetId', element: <DatasetDetailPage /> },
      { path: 'training/new', element: <TrainingPage /> },
      { path: 'runs', element: <RunsPage /> },
      { path: 'runs/:runId', element: <RunDetailPage /> },
      { path: 'models', element: <ModelsPage /> },
      { path: 'inference', element: <InferencePage /> },
      { path: '*', element: <NotFoundPage /> },
    ],
  },
]
