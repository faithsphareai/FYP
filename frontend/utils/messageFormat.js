// If this file doesn't exist, create it
export const formatMessage = (text, sender, options = {}) => {
  return {
    text,
    sender,
    timestamp: new Date().toISOString(),
    id: options.id || `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    isTyping: options.isTyping || false,
    ...options
  };
};
