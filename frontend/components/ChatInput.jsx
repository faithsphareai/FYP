import React, { useState } from 'react';
import { View, TextInput, TouchableOpacity, Text, StyleSheet } from 'react-native';

export const ChatInput = ({ onSend,  buttonLabel = 'Send'  }) => {
  const [message, setMessage] = useState('');

  const handleSend = () => {
    if (message.trim()) {
      onSend(message.trim());
      setMessage('');
    }
  };

  return (
    <View style={styles.inputContainer}>
      <TextInput
        style={styles.input}
        value={message}
        onChangeText={setMessage}
        placeholder="Type a message..."
        multiline
        maxHeight={100}
      />
      <TouchableOpacity 
        style={styles.sendButton}
        onPress={handleSend}
      >
        <Text style={styles.sendButtonText}>{buttonLabel}</Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  inputContainer: {
    flexDirection: 'row',
    padding: 16,
    borderTopWidth: 1,
    borderTopColor: '#E5E5EA',
    backgroundColor: 'white',
  },
  input: {
    flex: 1,
    borderWidth: 1,
    borderColor: '#E5E5EA',
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 8,
    marginRight: 8,
    maxHeight: 100,
  },
  sendButton: {
    backgroundColor: '#0b8c5c',
    borderRadius: 20,
    padding: 12,
    justifyContent: 'center',
  },
  sendButtonText: {
    color: 'white',
    fontWeight: '600',
  },
});