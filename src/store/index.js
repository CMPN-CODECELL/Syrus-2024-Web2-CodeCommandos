import { configureStore, createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import axios from "axios";
import { getDataRoute } from "../utils/api-routes";

const initialState = {
  userId: null,
  userData: null,
};

export const fetchUserData = createAsyncThunk(
  "user/fetchUserData",
  async (_, thunkAPI) => {
    try {
      const response = await axios.get(`${getDataRoute}`, {
        headers: {
          Authorization: localStorage.getItem("user"),
        },
      });
      console.log(response.data);
      return response.data;
    } catch (error) {
      throw error.response.data;
    }
  }
);

const userSlice = createSlice({
  name: "user",
  initialState,
  reducers: {
    setUserData: (state, action) => {
      state.userData = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder.addCase(fetchUserData.fulfilled, (state, action) => {
      state.userData = action.payload;
    });
  },
});

export const { setUserData } = userSlice.actions;

export const selectUserData = (state) => state.user.userData;

export const store = configureStore({
  reducer: {
    user: userSlice.reducer,
  },
});

export default store;
