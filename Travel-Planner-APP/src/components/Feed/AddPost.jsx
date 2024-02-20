import React, { useState, useCallback } from "react";
import styled from "styled-components";
import { FaUpload as Upload } from "react-icons/fa";
import { useDropzone } from "react-dropzone";
import { ToastContainer, toast } from "react-toastify";
import Post from "./Post";
export default function AddPost() {
  const [postPreview, setPostPreview] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [imagePreviews, setImagePreviews] = useState([]);
  const [text, setText] = useState("");
  const [postData, setPostData] = useState({
    mediaUrl: [],
    caption: "",
    username: "",
    userdp: "",
    likes: [0, []],
    comments: [0, []],
    shares: 0,
  });
  const toastOptions = {
    position: "bottom-left",
    autoClose: 4000,
    pauseOnHover: true,
    draggable: true,
    outerWidth: "fit-content",
    theme: "dark",
  };
  const handleFileChange = useCallback(
    (acceptedFiles) => {
      const newFiles = acceptedFiles.filter(
        (file) =>
          file.type.startsWith("image/") || file.type.startsWith("video/")
      );

      if (newFiles.length > 0 && newFiles.length <= 6) {
        setSelectedFiles([...selectedFiles, ...newFiles]);
        const newPreviews = newFiles.map((file) => URL.createObjectURL(file));
        setImagePreviews([...imagePreviews, ...newPreviews]);
      } else if (newFiles.length > 6) {
        toast.error("Cannot post more than 6 media", toastOptions);
      }
    },
    [selectedFiles, imagePreviews]
  );

  const { getRootProps, getInputProps } = useDropzone({
    onDrop: handleFileChange,
    accept: "image/*,video/*",
    multiple: true,
  });
  const handlePreview = () => {
    if (text != "" && selectedFiles.length != 0) {
      setPostPreview(true);
    } else {
      toast.error("Fill all fields", toastOptions);
    }
  };
  return (
    <>
      <Container>
        {!postPreview ? (
          <div id="addpost">
            <div className="post">
              <div {...getRootProps()} className="dropzone">
                <input {...getInputProps()} />
                {selectedFiles.length > 0 && (
                  <div className="image-preview">
                    <p>Selected Media Preview</p>
                    <div className="preview">
                      {selectedFiles.map((file, index) => (
                        <React.Fragment key={index}>
                          {file.type.startsWith("image/") ? (
                            <img
                              src={URL.createObjectURL(file)}
                              alt={`preview-${index}`}
                              className="logo"
                            />
                          ) : file.type.startsWith("video/") ? (
                            <video className="logo" controls>
                              <source
                                src={URL.createObjectURL(file)}
                                type={file.type}
                              />
                              Your browser does not support the video tag.
                            </video>
                          ) : null}
                        </React.Fragment>
                      ))}
                    </div>
                  </div>
                )}
                <p>
                  <Upload />
                  {`Upload ${selectedFiles.length > 0 ? "Other" : ""} Media`}
                </p>
                {selectedFiles.length > 0 && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedFiles([]);
                      setImagePreviews([]);
                    }}
                  >
                    Cancel Upload
                  </button>
                )}
              </div>
              <input
                type="text"
                name="caption"
                placeholder="Enter Your Post Caption"
                value={text}
                onChange={(e) => setText(e.target.value)}
              />
              <button onClick={handlePreview}>View Post Preview</button>
            </div>
          </div>
        ) : (
          <div className="posts">
            <div className="post2">
              <Post addpost={true} postData={postData} />
            </div>
          </div>
        )}
      </Container>
      <ToastContainer />
    </>
  );
}
const Container = styled.div`
  position: relative;
  height: 100%;
  width: 100%;
  color: white;
  padding: 0 1rem;
  padding-bottom: 1rem;
  #addpost {
    padding: 0.4rem;
    height: 100%;
    width: 100%;
    border-radius: 0.5rem;
    background-color: #282a34;
    .post {
      height: 100%;
      width: 100%;
      border-radius: 0.5rem;
      border: 0.1rem solid #4e3eff;
      padding: 1rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
      input {
        background-color: transparent;
        padding: 1rem;
        border: 0.1rem solid #4e0eff;
        border-radius: 0.4rem;
        color: white;
        width: 100%;
        font-size: 1rem;
        &:focus {
          border: 0.1rem solid #997af0;
          outline: none;
        }
      }
      button {
        align-self: center;
        background-color: #997af0;
        color: white;
        width: 100%;
        padding: 1rem;
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
      .dropzone {
        display: flex;
        gap: 0.5rem;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        border: 2px dashed #4e0eff;
        border-radius: 4px;
        background-color: #282a34;
        padding: 1rem;
        width: 100%;
        cursor: pointer;
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
          .preview {
            display: flex;
            align-items: center;
            justify-content: center;
            flex-wrap: wrap;
            gap: 1rem;
            img,
            video {
              height: 8rem;
              width: 8rem;
            }
          }
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
    }
  }
  .posts {
    height: 100%;
    width: 100%;
    padding-bottom: 1rem;
    align-items: stretch;
    .post2{
      height: 100%;
      background-color: #282a34;
      width: 100%;
      padding: 0.5rem;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      border-radius: 0.5rem;
      overflow-y: auto;
      &::-webkit-scrollbar {
        width: 0.2rem;
        height: 0.2rem;
        &-thumb {
          background-color: #ffffff39;
          width: 0.1rem;
          border-radius: 1rem;
        }
      }
    }
  }
`;
