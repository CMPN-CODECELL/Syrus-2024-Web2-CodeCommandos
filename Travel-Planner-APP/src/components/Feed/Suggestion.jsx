import React, { useState } from "react";
import styled from "styled-components";
import { FaSearch as Search, FaPlus } from "react-icons/fa";
import axios from "axios";
import { addSuggRoute, getSuggRoute } from "../../utils/api-routes";
export default function Suggestion() {
  const user = JSON.parse(localStorage.getItem("user"));
  const [city, setCity] = useState("");
  const [tag,setTag] = useState([])
  const [addS, seeaddS] = useState(false);
  const [sugg,setSugg] = useState([])
  const getSugg = async()=>{
    try {
      const {data} = await axios.post(getSuggRoute,{city,tag},{headers:{
        Authorization: user.token
      }})
      console.log(data);
      setSugg(data)
    } catch (error) {
      console.log(error.message);
    }
  }
  const handleClick = () => {};
  const [formData, setFormData] = useState({
    city: "",
    country: "",
    tag: "",
    msg: "",
  });
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: value,
    }));
  };
  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log(user.token);
    try {
      const { data } = await axios.post(addSuggRoute, formData, {
        headers: {
          Authorization: user.token,
        },
      });
      setFormData({ city: "", country: "", tag: "", msg: "" });
      console.log(data);
    } catch (error) {
      console.log(error.message);
    }
    console.log("Form data submitted:", formData);
  };
  return (
    <>
      {addS ? (
        <Container>
          <div className="top">
            <div className="search">
              <input
                type="text"
                placeholder="Suggestion Tag"
                value={tag}
                onChange={(e) => setTag(e.target.value)}
              />
            </div>
            <div className="search">
              <Search className="icon" onClick={getSugg} />
              <input
                type="text"
                placeholder="Suggestion City"
                value={city}
                onChange={(e) => setCity(e.target.value)}
              />
            </div>
            <div className="suggestions">
              {
                sugg.map((item)=>(
                  <div className="suggestion">
                    <h4>{`Tag: ${item.tag}`}</h4>
                    <h4>{`City: ${item.loc.city}`}</h4>
                    <h4>{`Message: ${item.msg}`}</h4>
                    <h4>{`Feedback: ${item.feedback}`}</h4>
                    <h4>{`At: ${item.createdAt.split("T")[0]}`}</h4>
                  </div>
                ))
              }
            </div>
            <div className="fab" onClick={() => seeaddS(true)}>
              <FaPlus className="fab-icon" />
            </div>
          </div>
        </Container>
      ) : (
        <FormContainer>
          <form onSubmit={handleSubmit}>
            <label>
              City:
              <input
                type="text"
                name="city"
                value={formData.city}
                onChange={handleChange}
              />
            </label>

            <label>
              Country:
              <input
                type="text"
                name="country"
                value={formData.country}
                onChange={handleChange}
              />
            </label>

            <label>
              Tag:
              <input
                type="text"
                name="tag"
                value={formData.tag}
                onChange={handleChange}
              />
            </label>

            <label>
              Message:
              <textarea
                name="msg"
                value={formData.msg}
                onChange={handleChange}
              />
            </label>
            <button type="submit">Add Suggestion</button>
          </form>
          <div className="fab" onClick={() => seeaddS(true)}>
            <FaPlus className="fab-icon" />
          </div>
        </FormContainer>
      )}
    </>
  );
}
const Container = styled.div`
  height: 100%;
  width: 100%;
  padding: 1rem;
  .top {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    .suggestions{
      height: 100%;
      width: 100%;
      display: flex;
      flex-direction: column;
      .suggestion{
        border-radius: 0.6rem;
        z-index: 2px;
        padding: 1rem;
        background-color: #282a34;
        h4{
          color: white;
        }
      }
    }
    .search {
      border-radius: 0.5rem;
      background-color: #373842;
      display: flex;
      flex-direction: row;
      justify-content: center;
      gap: 2rem;
      width: fit-content;
      input {
        outline: none;
        padding: 0.5rem;
        background-color: transparent;
        color: #7e7f89;
        border: none;
      }
      .icon {
        align-self: center;
        color: #595a63;
      }
    }
  }
  .fab {
    position: fixed;
    bottom: 2rem;
    right: 2rem;
    background-color: #007bff;
    color: #fff;
    border: none;
    border-radius: 50%;
    padding: 1rem;
    cursor: pointer;
    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);

    &:hover {
      background-color: #0056b3;
    }
  }

  .fab-icon {
    font-size: 1.5rem;
  }
`;
const FormContainer = styled.div`
.fab {
    position: fixed;
    bottom: 2rem;
    right: 2rem;
    background-color: #007bff;
    color: #fff;
    border: none;
    border-radius: 50%;
    padding: 1rem;
    cursor: pointer;
    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);

    &:hover {
      background-color: #0056b3;
    }
  }

  .fab-icon {
    font-size: 1.5rem;
  }
  width: 100%;
  height: 100%;
  margin: auto;
  padding: 20px;
  background-color: #282a34;
  border-radius: 8px;
  box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);

  form {
    display: grid;
    gap: 10px;

    label {
      display: block;
      font-weight: bold;
      color: white;
      font-size: medium;
    }

    input,
    textarea {
      background-color: #373842;
      color: white;
      width: 100%;
      padding: 8px;
      margin-top: 4px;
      border: 1px solid #ccc;
      border-radius: 4px;
      box-sizing: border-box;
    }

    button {
      background-color: #007bff;
      color: #fff;
      border: none;
      padding: 10px;
      border-radius: 4px;
      cursor: pointer;
    }
  }
`;
