import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import PropTypes from 'prop-types';

export const ChatHeader = ({ context, onNewTopic, onResetChat }) => {
  const router = useRouter();

  return (
    <View style={styles.header}>
      <TouchableOpacity onPress={() => router.back()}>
        <Ionicons name="arrow-back" size={24} color="#15803d" />
      </TouchableOpacity>

      <Text style={styles.title}>{context}</Text>

      {context === 'Islamic Quiz' && (
        <TouchableOpacity onPress={onNewTopic} style={styles.newTopicButton}>
          <Text style={styles.newTopicText}>New Topic</Text>
        </TouchableOpacity>
      )}

      {/* Reset icon button */}
      <TouchableOpacity onPress={onResetChat}  className="p-2 rounded-full bg-green-50">
        <Ionicons name="refresh" size={24} color="#15803d" />
      </TouchableOpacity>
    </View>
  );
};

ChatHeader.propTypes = {
  context: PropTypes.string.isRequired,
  onNewTopic: PropTypes.func,
  onResetChat: PropTypes.func, // Required for reset button
};

const styles = StyleSheet.create({
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e5e5',
    backgroundColor: '#fff',
  },
  title: {
    flex: 1,
    fontSize: 18,
    fontWeight: '600',
    color: '#000',
    marginLeft: 16,
  },
  newTopicButton: {
    backgroundColor: '#0b8c5c',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    marginRight: 12,
  },
  newTopicText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '500',
  },
});
