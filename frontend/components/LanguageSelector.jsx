// components/LanguageSelectorModal.js
import React from 'react';
import {
  Modal,
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
} from 'react-native';
import { useChat } from '../context/ChatContext';

const LanguageSelectorModal = ({ visible, onClose }) => {
  const { setLanguage } = useChat();

  const chooseLanguage = (lang) => {
    setLanguage(lang);
    onClose(); // Close modal after selection
  };

  return (
    <Modal visible={visible} animationType="fade" transparent>
      <View style={styles.overlay}>
        <View style={styles.modal}>
          <Text style={styles.title}>🌐 Select Your Language</Text>

          <TouchableOpacity
            style={[styles.button, styles.englishButton]}
            onPress={() => chooseLanguage('en')}
          >
            <Text style={styles.buttonText}>English</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.button, styles.arabicButton]}
            onPress={() => chooseLanguage('ar')}
          >
            <Text style={styles.buttonText}>Arabic (العربية)</Text>
          </TouchableOpacity>

          <TouchableOpacity onPress={onClose} style={styles.closeButton}>
            <Text style={styles.closeText}>Cancel</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );
};

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.6)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modal: {
    width: '85%',
    backgroundColor: '#fff',
    borderRadius: 20,
    paddingVertical: 30,
    paddingHorizontal: 20,
    alignItems: 'center',
    elevation: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
  },
  title: {
    fontSize: 20,
    fontWeight: '600',
    marginBottom: 24,
    color: '#333',
  },
  button: {
    width: '100%',
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
    marginBottom: 16,
  },
  englishButton: {
    backgroundColor: '#4CAF50',
  },
  arabicButton: {
    backgroundColor: '#2196F3',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '500',
  },
  closeButton: {
    marginTop: 8,
  },
  closeText: {
    color: '#999',
    fontSize: 14,
    textDecorationLine: 'underline',
  },
});

export default LanguageSelectorModal;
