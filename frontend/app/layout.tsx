import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'AI Driver Safety & Intelligence Platform',
  description:
    'Analyze dashcam driving footage with computer vision and deep learning. ' +
    'Detect road objects, understand your driving environment, and receive an ' +
    'intelligent risk assessment powered by YOLO, PyTorch, and XGBoost.',
  keywords: [
    'AI driver safety',
    'dashcam analysis',
    'computer vision',
    'driving risk assessment',
    'object detection',
    'YOLO',
    'deep learning',
  ],
  authors: [{ name: 'AI Driver Safety Platform' }],
  openGraph: {
    title: 'AI Driver Safety & Intelligence Platform',
    description: 'Intelligent dashcam analysis powered by computer vision and deep learning.',
    type: 'website',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body className="bg-noise">
        {children}
      </body>
    </html>
  );
}
