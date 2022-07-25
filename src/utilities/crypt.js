// https://dannydenenberg.com/end-to-end-nodejs
var CryptoJS = require('crypto-js');

const secret_key_from_env = 'Hrh0SRVifRpUaiClvaL05qStuorOUPPUz18Z7lKS';

export function encrypt(message, chat_room_name) {
  const key = chat_room_name + secret_key_from_env;
  return CryptoJS.AES.encrypt(message, key).toString();
}

export function decrypt(message, chat_room_name) {
  const key = chat_room_name + secret_key_from_env;
  return CryptoJS.AES.decrypt(message, key).toString(CryptoJS.enc.Utf8);
}
