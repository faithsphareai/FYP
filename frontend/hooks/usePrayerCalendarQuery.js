import { useQuery } from '@tanstack/react-query';
import { getPrayerCalendar } from '../utils/services/apis';

/**
 * Hook to fetch monthly prayer timings
 * @param {Object} coords - User coordinates with latitude and longitude
 * @param {number} year 
 * @param {number} month 
 * @param {number} method 
 * @param {number} school 
 */
export const usePrayerCalendarQuery = (coords, year, month, method = 1, school = 0) => {
  return useQuery({
    queryKey: ['prayer-calendar', coords, year, month, method, school],
    queryFn: () => getPrayerCalendar(coords, year, month, method, school),
    enabled: !!coords,
  });
};
