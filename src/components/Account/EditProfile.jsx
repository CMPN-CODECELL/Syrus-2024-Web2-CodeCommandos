import React, { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import styled from "styled-components";
import { ToastContainer, toast } from "react-toastify";
import { FaUpload as Upload } from "react-icons/fa";
import Back from "./Back";
import { isEqual } from "lodash";
export default function EditProfile({ changeMenu, user }) {
  const [visible, setIsVisible] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [texts, setTexts] = useState({
    name: user.name,
    username: user.username,
    picture: user.picture,
    country: user.country,
    city: user.city,
  });
  const [edited, setEdited] = useState(false);
  const toastOptions = {
    position: "bottom-left",
    autoClose: 4000,
    pauseOnHover: true,
    draggable: true,
    outerWidth: "fit-content",
    theme: "dark",
  };
  const handleFileChange = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file && file.type.startsWith("image/")) {
      setSelectedFile(file);
      const previewURL = URL.createObjectURL(file);
      setImagePreview(previewURL);
      setEdited(true);
    } else {
      toast.error("Please select a Image file", toastOptions);
      setSelectedFile(null);
      setImagePreview(null);
    }
  }, []);
  const handleTextChange = (e) => {
    setTexts({ ...texts, [e.target.name]: e.target.value });
  };
  const { getRootProps, getInputProps } = useDropzone({
    onDrop: handleFileChange,
    accept: "image/*",
    multiple: false,
  });
  const saveChanges = () => {
    if (isEqual(texts, user) && !edited) {
      toast.error("Edit any field to save changes", toastOptions);
    }
  };
  return (
    <>
      <Container visible={visible}>
        <div className="top-bar">
          <div className="back">
            <Back changeMenu={changeMenu} />
          </div>
          <div className="text">
            <h3>Edit Profile</h3>
          </div>
        </div>
        <div className="form">
          <form onSubmit={(e) => e.preventDefault()}>
            <img
              src={texts.picture? `http://localhost:5000/${texts.picture}`: ''}
              alt=""
              className="logo"
              onClick={() => {
                setIsVisible(!visible);
              }}
            />
            {visible && (
              <div {...getRootProps()} className="dropzone">
                <input {...getInputProps()} />
                {selectedFile && (
                  <div className="image-preview">
                    <p>Selected Image Preview</p>
                    <img
                      src={imagePreview}
                      alt="dp"
                      className="logo"
                      onClick={() => {
                        setIsVisible(!visible);
                      }}
                    />
                  </div>
                )}
                <p>
                  <Upload />
                  {`Upload ${selectedFile ? "Other" : ""} Profile Picture`}
                </p>
                {selectedFile && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setIsVisible(false);
                      setSelectedFile(null);
                      setEdited(false);
                    }}
                  >
                    Cancel Upload
                  </button>
                )}
              </div>
            )}
            <label htmlFor="name">Name: </label>
            <input
              type="text"
              name="name"
              id="name"
              placeholder="Name"
              value={texts.name}
              onChange={(e) => handleTextChange(e)}
            />
            <label htmlFor="username">Username:</label>
            <input
              type="text"
              name="username"
              id="username"
              placeholder="Username"
              value={texts.username}
              onChange={(e) => handleTextChange(e)}
            />
            <label htmlFor="Country">Country</label>
            <input
              type="text"
              name="country"
              id="country"
              placeholder="Country"
              value={texts.country ? texts.country : ''}
              onChange={(e) => handleTextChange(e)}
            />
            <label htmlFor="City">Country</label>
            <input
              type="text"
              name="City"
              id="City"
              placeholder="City"
              value={texts.city ? texts.city : ''}
              onChange={(e) => handleTextChange(e)}
            />
            <button onClick={saveChanges}>Save</button>
          </form>
        </div>
      </Container>
      <ToastContainer />
    </>
  );
}

const Container = styled.div`
  user-select: none;
  height: 100vh;
  width: 100%;
  display: grid;
  grid-template-rows: 5% 95%;
  padding: 1rem;
  background-color: #282a34;
  .top-bar {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 1rem;
    padding-bottom: 1rem;
    .text {
      h3 {
        color: white;
      }
    }
  }
  .form {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    form {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      img {
        align-self: center;
        height: 6rem;
        width: 6rem;
        border-radius: 3rem;
        padding: 0.2rem;
        border: 0.1rem solid #4e0eff;
      }
      .dropzone {
        display: flex;
        gap: 0.5rem;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        border: 2px dashed #4e0eff;
        border-radius: 4px;
        background-color: #282a34;
        padding: 0.5rem;
        cursor: pointer;
        margin-bottom: 20px;
        .logo {
          height: 100px;
          width: 100px;
          margin-bottom: 10px;
        }
        p {
          color: white;
        }
        .image-preview {
          display: flex;
          flex-direction: column;
          gap: 0.4rem;
          p {
            color: white;
            align-self: center;
          }
        }
        button {
          align-self: center;
          background-color: #997af0;
          color: white;
          width: fit-content;
          padding: 0.3rem 0.3rem;
          border: none;
          font-weight: bold;
          cursor: pointer;
          border-radius: 0.4rem;
          font-size: 1rem;
          text-transform: uppercase;
          transition: 0.5s ease-in-out;
          &:hover {
            background-color: #4e0eff;
          }
        }
      }
      label {
        font-size: large;
        color: white;
      }
      input {
        border: none;
        outline: none;
        background-color: transparent;
        color: white;
      }
      .radio {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        .public,
        .private {
          display: flex;
          flex-direction: row;
          gap: 1rem;
          label {
            font-size: medium;
          }
        }
      }
      button {
        align-self: center;
        background-color: #997af0;
        color: white;
        width: fit-content;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: bold;
        cursor: pointer;
        border-radius: 0.4rem;
        font-size: 1rem;
        text-transform: uppercase;
        transition: 0.5s ease-in-out;
        &:hover {
          background-color: #4e0eff;
        }
      }
    }
  }
`;
