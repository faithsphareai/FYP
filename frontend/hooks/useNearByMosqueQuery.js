import { useQuery } from '@tanstack/react-query';
import { getNearByMosques } from '../utils/services/apis';

/**
 * Hook to fetch nearby mosques
 * @param {Object} coords - User coordinates object with latitude and longitude
 * @param {number} limit - Maximum number of results to return (default: 10)
 * @param {number} zoom - Map zoom level (default: 10)
 * @returns {QueryResult} React Query result object
 */
export const useNearbyMosquesQuery = (coords, limit = 10, zoom = 10) => {
  return useQuery({
    queryKey: ['nearby-mosques', coords, limit, zoom],
    queryFn: () => getNearByMosques(coords, limit, zoom),
    enabled: !!coords,
    staleTime: 1000 * 60 * 60, // 1 hour stale time
    cacheTime: 1000 * 60 * 60 * 2, // 2 hour cache time
    retry: 2,
    retryDelay: 1000,
  });
};