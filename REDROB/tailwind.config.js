export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        primary: '#005243',
        'primary-fixed': '#a6f1db',
        'primary-fixed-dim': '#8ad5c0',
        secondary: '#5846c8',
        tertiary: '#8b5200',
        surface: '#f7faf7',
        'surface-container': '#ecefec',
        'surface-container-high': '#e6e9e6',
        'on-surface': '#0a0c0b',
        'on-surface-variant': '#3f4945',
        outline: '#6f7975'
      },
      borderRadius: {
        DEFAULT: '0.125rem',
        lg: '0.25rem',
        xl: '0.5rem'
      },
      fontFamily: {
        display: ['Inter', 'ui-sans-serif', 'system-ui'],
        body: ['Inter', 'ui-sans-serif', 'system-ui'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'monospace']
      }
    }
  },
  plugins: []
}
