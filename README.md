
# Smart Document Knowledge Base  
**Student Name:** Jesus Amos Facundo  
**Date:** July 3, 2025  

---

## Features Implemented

### Day 2 Features:
**My chosen features:**

1. **Source Attribution** – The Q&A tab displays the source document from which the answer was retrieved, providing transparency and traceability.  
2. **Tabbed Interface** – The application is organized into clear, functional tabs (Document Upload, Q&A, Summaries, Search History), improving navigation and user experience.  
3. **Search History** – Each user question is saved in a dedicated tab, allowing users to revisit past queries in an organized manner.  

---

### Day 3 Styling:
**My chosen styling:**

1. **Color Themes** – Custom background using a `background.jpg` image, combined with coordinated font and button colors for a cohesive and calm blue-themed interface.  
2. **Loading Animations** – Added spinners to the document upload and summarization steps, enhancing user feedback during processing.  
3. **Layout Improvements** – Centered the app layout, increased font readability, balanced spacing, and visually separated sections with consistent headers.  

---

## How to Run

1. Install required packages:  
```bash
pip install streamlit chromadb transformers torch docx pdfminer.six
```

2. Run the app:  
```bash
streamlit FileTalkApp_final.py
```

3. Upload `.txt`, `.pdf`, or `.docx` files and start asking questions!

---

## Challenges & Solutions

- **Styling the Streamlit app with a background image**  
  *Challenge:* Streamlit doesn’t support background images natively.  
  *Solution:* Injected custom CSS using `st.markdown()` and verified file path alignment to ensure the background rendered.

- **ChromaDB version conflict**  
  *Challenge:* Deprecated configuration caused app to crash.  
  *Solution:* Updated to the new `PersistentClient` configuration based on migration docs and replaced legacy instantiation.

- **Managing tab content and structure**  
  *Challenge:* Adding new tabs without breaking existing functionality.  
  *Solution:* Refactored functions and used `st.session_state` to store and control cross-tab behavior.

---

## What I Learned

This project helped me learn how to build a modular and visually appealing Streamlit app using external libraries like ChromaDB and Transformers. I gained practical experience with frontend styling using CSS in Python apps and deepened my understanding of how to structure a user-friendly AI-powered document assistant. Most importantly, I improved my debugging and feature-testing workflow to avoid breaking code while implementing new functionality.
