import { StyleSheet } from 'react-native';

export const prayerStyles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#ffffff',
  },
  navigation: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 8,
  },
  navButton: {
    padding: 8,
  },
  disabledButton: {
    opacity: 0.5,
  },
  stepText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#15803d',
  },
  scrollView: {
    flex: 1,
    paddingHorizontal: 16,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#15803d',
    marginBottom: 8,
  },
  image: {
    height: 200,
    borderRadius: 12,
    marginBottom: 16,
  },
  descriptionContainer: {
    backgroundColor: '#f0fdf4',
    borderRadius: 8,
    padding: 16,
    marginBottom: 16,
  },
  descriptionText: {
    fontSize: 18,
    color: '#15803d',
  },
  detailsContainer: {
    marginBottom: 24,
  },
  detailRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  bullet: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#16a34a',
    marginRight: 8,
  },
  detailText: {
    fontSize: 16,
    color: '#374151',
  },
  progressContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 16,
    gap: 8,
  },
  progressDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#d1d5db',
  },
  activeDot: {
    backgroundColor: '#16a34a',
  },
});