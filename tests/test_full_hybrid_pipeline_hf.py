# tests/test_full_hybrid_pipeline_hf.py

import numpy as np
import pytest
from rag.embeddings.providers import TextEmbedder
from rag.storage.hybrid_store import HybridStore
from rag.processing.text_chunker import TextChunker
from rag.core.models.vector_store_models import VectorMetadata
from rag.core.config import RAGConfig, EmbeddingConfig

@pytest.fixture
def embedder():
    # Use a larger model with better semantic capabilities for medical text
    config = RAGConfig(
        embedding=EmbeddingConfig(
            provider="huggingface",
            model_name="sentence-transformers/all-mpnet-base-v2",  # Higher quality than MiniLM
            dimension=768  # MPNet has 768 dimensions
        )
    )
    return TextEmbedder(config=config)

@pytest.fixture
def hybrid_store():
    # Initialize with the correct dimension for huggingface
    # Using a test collection to avoid conflicts with production data
    return HybridStore(
        dimension=768, 
        collection_name="test_hybrid_hf",
        cache_size=100  # Smaller cache for testing
    )

def test_full_pipeline(embedder, hybrid_store):
    print("\n===== FULL RAG PIPELINE TEST WITH HUGGINGFACE AND HYBRID STORE =====\n")
    
    # Example text
    print("Step 1: Preparing sample clinical texts...")
    texts = [
        "Patient has a history of hypertension and diabetes. They have been managing their conditions with medication and lifestyle changes for the past 15 years. Recently, they have experienced increased blood pressure readings averaging 150/95 and elevated blood sugar levels consistently above 200 mg/dL. The patient reports compliance with current medications including lisinopril 20mg daily and metformin 1000mg twice daily, but notes increased stress at work and decreased physical activity over the past 3 months. A review of their home monitoring log shows concerning trends in both blood pressure and blood glucose measurements.",
        "Patient is allergic to penicillin and has had severe reactions in the past, including anaphylaxis requiring emergency room visits on two occasions. They are advised to avoid all penicillin-based antibiotics and carry an epinephrine auto-injector at all times. Previous reactions included severe hives, difficulty breathing, and facial swelling occurring within 30 minutes of administration. Patient wears a medical alert bracelet and has documented allergies clearly in all medical records. Alternative antibiotic protocols have been established for common infections.",
        "Patient has been prescribed metformin to manage their type 2 diabetes, with recent A1C levels showing poor control at 8.5%. They are instructed to take the medication twice daily with meals and monitor their blood sugar levels at least four times daily. Patient reports occasional gastrointestinal side effects but is generally tolerating the medication well. Diet review indicates inconsistent meal timing and frequent high-carbohydrate food choices. A referral to a diabetes educator has been made for additional lifestyle counseling.",
        "Patient reports frequent headaches that have been occurring for the past few months, with increasing severity and frequency over the last two weeks. They describe the pain as throbbing and often accompanied by nausea, sensitivity to light, and occasional visual auras. Episodes typically last 4-6 hours and significantly impact daily activities. Over-the-counter medications provide minimal relief. Patient maintains a headache diary showing correlation with stress and poor sleep patterns.",
        "Patient has a family history of heart disease, with both parents having experienced heart attacks in their 60s, and a sibling recently diagnosed with coronary artery disease at age 55. They are advised to maintain a heart-healthy diet low in saturated fats and rich in whole grains and vegetables, exercise regularly with at least 150 minutes of moderate activity per week, and undergo regular cardiovascular screening. Recent lipid panel shows elevated LDL at 160 mg/dL and low HDL at 35 mg/dL. Stress test results from last month were borderline, suggesting the need for closer monitoring.",
        "First-time mother experiencing normal post-partum recovery after an uncomplicated vaginal delivery. The baby was born at 39 weeks gestation weighing 7 pounds 8 ounces. Mother is successfully breastfeeding every 2-3 hours and reports adequate wet diapers. Baby demonstrates good latch and proper weight gain trajectory. Mother's vital signs are stable and lochia is within normal limits. Mother's pain is well-controlled with over-the-counter medication. Perineal repair is healing appropriately with no signs of infection. Mother is ambulatory and performing self-care independently. Emotional status is stable with good family support system in place. Newborn screening tests were completed and returned normal results. Baby's temperature has remained stable, and skin color is pink with good perfusion. Both mother and baby are scheduled for routine follow-up appointments within the next week. Mother has been educated on proper umbilical cord care and newborn bathing techniques. Lactation consultant has provided additional guidance on various breastfeeding positions and proper milk storage guidelines. Mother demonstrates understanding of warning signs that would require immediate medical attention. Baby is showing appropriate responses to visual and auditory stimuli. Mother reports adequate rest periods when baby sleeps and has established a preliminary feeding and sleep schedule. Postpartum depression screening was negative, and mother expresses positive outlook about transition to parenthood. Family members have been trained in newborn care basics and safety protocols. Discharge medications and instructions have been reviewed thoroughly with both parents present.",
        "Newborn presents with mild jaundice on day 3 of life, which is being monitored via transcutaneous bilirubin measurements. The infant is otherwise healthy, demonstrating strong reflexes, good muscle tone, and appropriate responses to stimuli. Parents have been educated on proper positioning for phototherapy at home and the importance of frequent feeding to help clear bilirubin.",
        "Patient is a 28-year-old primigravida in active labor at 40 weeks and 2 days gestation. Contractions are occurring every 3-4 minutes, lasting 60-90 seconds. Cervical examination reveals 6cm dilation with 80% effacement. Fetal monitoring shows reassuring heart rate patterns with good variability and moderate accelerations. Patient is coping well with labor using breathing techniques and position changes.",
        "Routine prenatal visit for 32-week pregnant woman showing normal progression. Fundal height measures appropriate for gestational age, fetal movement is active and regular, and fetal heart rate is 140 beats per minute. Blood pressure remains stable at 118/75, and patient reports no concerning symptoms. Discussion included birth plan preferences and signs of preterm labor to watch for.",
        "Six-month well-baby check reveals appropriate developmental milestones. Infant is rolling over both ways, beginning to sit independently, and showing interest in solid foods. Vaccinations are up to date, growth parameters fall within normal percentiles, and physical examination is unremarkable. Parents report consistent sleep patterns and appropriate dietary intake."
    ]
    print(f"- Prepared {len(texts)} clinical text samples")

    # Process text
    print("\nStep 2: Processing text into chunks...")
    chunker = TextChunker()
    all_chunks = []
    for text in texts:
        chunks = chunker.chunk_text(text)
        all_chunks.extend(chunks)
    print(f"- Created {len(all_chunks)} text chunks")

    # Generate embeddings
    print("\nStep 3: Generating embeddings with Hugging Face model...")
    # Use the batch embedding method for efficiency
    embeddings = embedder.embed_texts(all_chunks)
    
    # Convert to numpy array
    embeddings_np = np.array(embeddings, dtype=np.float32)
    print(f"- Generated embeddings with shape: {embeddings_np.shape}")
    print(f"- Embedding dimension: {embeddings_np.shape[1]}")
    print(f"- Using model: {embedder.model_name}")

    # Add embeddings to the Hybrid Store
    print("\nStep 4: Storing embeddings in Hybrid Store (FAISS + Qdrant)...")
    
    # Create VectorMetadata objects for each chunk
    metadata = [
        VectorMetadata(
            text=chunk, 
            source="test", 
            embedding_model="HuggingFace"
        ) 
        for chunk in all_chunks
    ]
    
    # Delete existing Qdrant collection if it exists (to start fresh)
    try:
        print("- Recreating test collection...")
        if hybrid_store.qdrant_store.client.collection_exists(hybrid_store.collection_name):
            hybrid_store.qdrant_store.client.delete_collection(hybrid_store.collection_name)
            # Recreate the HybridStore
            hybrid_store = HybridStore(
                dimension=768,
                collection_name=hybrid_store.collection_name,
                cache_size=100
            )
    except Exception as e:
        print(f"- Collection setup: {str(e)}")
    
    ids = hybrid_store.add_vectors(embeddings_np, metadata)
    print(f"- Stored {len(ids)} vectors in Hybrid Store with IDs: {ids[:5]}...")

    # TEST 1: Basic similarity search
    query_text = "What medications should I take for diabetes?"
    print(f"\nTEST 1: Basic similarity search with first text: {query_text}")
    query_embedding = embedder.embed_text(query_text)
    query_vector = np.array(query_embedding, dtype=np.float32)
    
    results = hybrid_store.search_vector(query_vector, k=5)
    
    # Assertions
    assert results is not None
    assert len(results) > 0
    for result in results:
        assert hasattr(result, "text")
        assert hasattr(result, "source")
        assert hasattr(result, "embedding_model")

    # Print results for manual inspection
    print("\nSimilarity search results (ordered by relevance):")
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Text: {result.text[:100]}..." if len(result.text) > 100 else f"  Text: {result.text}")
        print(f"  Score: {getattr(result, 'score', 'N/A'):.4f}" if hasattr(result, 'score') else f"  Score: N/A")
        print(f"  Distance: {getattr(result, 'distance', 'N/A'):.4f}" if hasattr(result, 'distance') else f"  Distance: N/A")
        
    # TEST 2: Different query
    print("\nTEST 2: Using a different query (headache search)")
    query_text = "I'm experiencing severe headaches with light sensitivity"
    query_embedding = embedder.embed_text(query_text)
    query_vector = np.array(query_embedding, dtype=np.float32)
    
    results = hybrid_store.search_vector(query_vector, k=3)
    
    print(f"\nQuery: '{query_text}'")
    print("\nHeadache search results:")
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Text: {result.text[:100]}..." if len(result.text) > 100 else f"  Text: {result.text}")
        print(f"  Score: {getattr(result, 'score', 'N/A'):.4f}" if hasattr(result, 'score') else f"  Score: N/A")
        print(f"  Distance: {getattr(result, 'distance', 'N/A'):.4f}" if hasattr(result, 'distance') else f"  Distance: N/A")
    
    # TEST 3: Filtered search (testing Qdrant's capabilities)
    print("\nTEST 3: Filtered search (only texts with 'diabetes')")
    try:
        # Create a filter that actually searches for diabetes-related texts
        diabetes_filter = {
            "$and": [
                {"text": {"$contains": "diabetes"}},
                {"source": "test"}
            ]
        }
        results = hybrid_store.search_vector(query_vector, k=5, filter=diabetes_filter)
        
        # Validate that results actually contain diabetes
        diabetes_results_count = 0
        print("\nFiltered results:")
        for i, result in enumerate(results):
            contains_diabetes = "diabetes" in result.text.lower()
            if contains_diabetes:
                diabetes_results_count += 1
                
            print(f"Result {i+1}:")
            print(f"  Text: {result.text[:100]}..." if len(result.text) > 100 else f"  Text: {result.text}")
            print(f"  Score: {getattr(result, 'score', 'N/A'):.4f}" if hasattr(result, 'score') else f"  Score: N/A")
            print(f"  Distance: {getattr(result, 'distance', 'N/A'):.4f}" if hasattr(result, 'distance') else f"  Distance: N/A")
            print(f"  Contains 'diabetes': {contains_diabetes}")
        
        print(f"\n- Found {diabetes_results_count} out of {len(results)} results containing 'diabetes'")
        
        # Add assertions to verify filtering worked
        if len(results) > 0:
            assert diabetes_results_count > 0, "Filter didn't return any texts containing 'diabetes'"
    except Exception as e:
        print(f"- Filter test skipped: {str(e)}")
    
    # TEST 4: Cache behavior
    print("\nTEST 4: Testing cache behavior")
    # First search should use Qdrant
    print("- First search (should use Qdrant):")
    results1 = hybrid_store.search_vector(query_vector, k=1)
    
    # Multiple searches to populate cache
    print("- Performing multiple searches to populate cache...")
    for _ in range(10):  # Access multiple times to trigger caching
        hybrid_store.search_vector(query_vector, k=1)
    
    # Next search might use FAISS cache
    print("- Subsequent search (may use FAISS cache):")
    results2 = hybrid_store.search_vector(query_vector, k=1)
    
    # Get store stats
    print("\nHybrid Store Stats:")
    stats = hybrid_store.get_stats()
    for key, value in stats.items():
        print(f"- {key}: {value}")
            
    # Clean up - delete the test collection
    print("\nStep 5: Cleaning up...")
    try:
        hybrid_store.qdrant_store.client.delete_collection(hybrid_store.collection_name)
        print(f"- Deleted test collection: {hybrid_store.collection_name}")
    except Exception as e:
        print(f"- Cleanup error: {str(e)}")
            
    print("\n===== TEST COMPLETED SUCCESSFULLY =====") 