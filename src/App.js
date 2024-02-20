import React from "react";
import {BrowserRouter,Routes,Route} from 'react-router-dom'
import Home from "./pages/Home";
import Auth from "./pages/Auth";
function App() {
  return (
    <BrowserRouter>
    <Routes>
      <Route exact path="/auth" element={<Auth/>}/>
      <Route exact path="/" element={<Home/>}/>
    </Routes>
    </BrowserRouter>
  );
}

export default App;
