import { createSlice } from '@reduxjs/toolkit';

//createslice is  a reduxjstoolkit api that accepts an initial state, an object of reducer functions, and a "slice name", and automatically generates action creators and action types that correspond to the reducers and state.

export const userSlice = createSlice({
  name: 'user',
  initialState: {
    user: null,
  },
  //Reducers are functions that take the current state and an action as arguments, and return a new state result. In other words, (state, action) => newState.

  //An action is just an object that always contains the type of logic that is going to be executed on the state and the payload or data that is coming from the action.

  reducers: {
    login: (state, action) => {
      state.user = action.payload; //payload stores the data we want to pass here payload stores current state of the user
    },
    logout: (state) => {
      state.user = null;
    },
  },
});

export const { login, logout } = userSlice.actions;

export const selectUser = (state) => state.user.user;
export default userSlice.reducer;
