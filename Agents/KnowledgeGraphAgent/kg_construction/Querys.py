import re

from Agents.KnowledgeGraphAgent import graphdb


def test_match_query():
    # first, take a look at the entity labels
    results = graphdb.query("""MATCH (n)
        WHERE n:`__Entity__`
        RETURN DISTINCT labels(n) AS entity_labels
        """)

    print(results['query_result'])


def test_unwind_labels():
    # unwind those lists of labels
    results = graphdb.query("""MATCH (n)
        WHERE n:`__Entity__`
        WITH DISTINCT labels(n) AS entity_labels
        UNWIND entity_labels AS entity_label
        RETURN DISTINCT entity_label
        """)

    print(results['query_result'])

def test_filter_labels():
    # filter out labels that start with "__"
    results = graphdb.query("""MATCH (n)
        WHERE n:`__Entity__`
        WITH DISTINCT labels(n) AS entity_labels
        UNWIND entity_labels AS entity_label
        WITH entity_label
        WHERE NOT entity_label STARTS WITH "__"
        RETURN entity_label
        """)

    print(results['query_result'])

# wrap the query into a callable function
def find_unique_entity_labels():
    result = graphdb.query("""MATCH (n)
        WHERE n:`__Entity__`
        WITH DISTINCT labels(n) AS entity_labels
        UNWIND entity_labels AS entity_label
        WITH entity_label
        WHERE NOT entity_label STARTS WITH "__"
        RETURN collect(entity_label) as unique_entity_labels
        """)
    if result['status'] == 'error':
        raise Exception(result['message'])
    return result['query_result'][0]['unique_entity_labels']




def find_unique_entity_keys(entityLabel: str):
    result = graphdb.query("""MATCH (n:$($entityLabel))
    WHERE n:`__Entity__`
    WITH DISTINCT keys(n) as entityKeys
    UNWIND entityKeys as entityKey
    RETURN collect(distinct(entityKey)) as unique_entity_keys
    """, {
        "entityLabel": entityLabel
    })
    if result['status'] == 'error':
        raise Exception(result['message'])
    return result['query_result'][0]['unique_entity_keys']



def find_unique_domain_keys(domainLabel: str):
    result = graphdb.query("""MATCH (n:$($domainLabel))
    WHERE NOT n:`__Entity__` // exclude entities created by the KG builder, these should be domain nodes
    WITH DISTINCT keys(n) as domainKeys
    UNWIND domainKeys as domainKey
    RETURN collect(distinct(domainKey)) as unique_domain_keys
    """, {
        "domainLabel": domainLabel
    })
    if result['status'] == 'error':
        raise Exception(result['message'])
    return result['query_result'][0]['unique_domain_keys']




def normalize_key(label:str, key:str) -> str:
    """Normalizes a a property key for a given label.

    Keys are normalized by:
    - lowercase the key
    - remove any leading/trailing whitespace
    - remove label prefix from key
    - replace internal whitespace with "_"

    for example:
        - "Product_name" -> "name"
        - "product name" -> "name"
        - "price" -> "price

    Args:
        label (str): The label to normalize keys for
        keys (List[str]): The list of keys to normalize

    Returns:
        List[str]: The normalized list of keys
    """
    lowercase_key = key.lower()
    unprefixed_key = re.sub(f"^{label.lower()}[_ ]*", "", lowercase_key)
    normalized_key = re.sub(" ", "_", unprefixed_key)
    return normalized_key



# use the rapidfuzz library for fuzzy text similarity scoring
from rapidfuzz import fuzz

# for a given label, get pairs of entity and domain keys that correlate
def correlate_entity_and_domain_keys(label: str, entity_keys: list[str], domain_keys: list[str], similarity: float = 0.9) -> list[tuple[str, str]]:
    correlated_keys = []
    for entity_key in entity_keys:
        for domain_key in domain_keys:
            # only consider exact matches. this could use fuzzy matching
            normalized_entity_key = normalize_key(label, entity_key)
            normalized_domain_key = normalize_key(label, domain_key)
            # rapidfuzz similarity is 0.0 -> 100.0, so divide by 100 for 0.0 -> 1.0
            fuzzy_similarity = (fuzz.ratio(normalized_entity_key, normalized_domain_key) / 100)
            if (fuzzy_similarity > similarity):
                correlated_keys.append((entity_key, domain_key, fuzzy_similarity))
    correlated_keys.sort(key=lambda x: x[2], reverse=True)
    return correlated_keys

def find_similar_entity_values(entityLabel: str, entityKey: str, domainLabel: str, domainKey: str):
    # use the Jaro-Winkler function to calculate distance between product names
    results = graphdb.query("""
    // MATCH all pairs of subject and domain nodes -- this is an expensive cartesian product
    MATCH (entity:$($entityLabel):`__Entity__`), (domain:$($entityLabel))
    WITH entity, domain, apoc.text.jaroWinklerDistance(entity[$entityKey], domain[$domainKey]) as score
    // experiment with different thresholds to see how the results change
    WHERE score < 0.4
    RETURN entity[$entityKey] AS entityValue, domain[$domainKey] AS domainValue, score
    // experiment with different limits to see more or fewer pairs
    LIMIT 3
    """, {
        "entityLabel": "Product",
        "entityKey": "name",
        "domainKey": "product_name"
    })

    print(results['query_result'])


def find_similar_entity_values(entityLabel: str, entityKey: str, domainLabel: str, domainKey: str):
    # connect all corresponding nodes with a relationship
    results = graphdb.query("""
    MATCH (entity:$($entityLabel):`__Entity__`),(domain:$($entityLabel))
    // use the score as a predicate to filter the pairs. this is better
    WHERE apoc.text.jaroWinklerDistance(entity[$entityKey], domain[$domainKey]) < 0.1
    MERGE (entity)-[r:CORRESPONDS_TO]->(domain)
    ON CREATE SET r.created_at = datetime()
    ON MATCH SET r.updated_at = datetime()
    RETURN elementId(entity) as entity_id, r, elementId(domain) as domain_id
    """, {
        "entityLabel": "Product",
        "entityKey": "name",
        "domainKey": "product_name"
    })

    print(results['query_result'])

    # run this repeatedly to illustrate that MERGE only happens once


# wrap as a function
def correlate_subject_and_domain_nodes(label: str, entity_key: str, domain_key: str, similarity: float = 0.9) -> dict:
    """Correlate entity and domain nodes based on label, entity key, and domain key,
    where the corresponding values of the entity and domain properties are similar

    For example, if you have a label "Person" and an entity key "name", and a domain key "person_name",
    this function will create a relationship like:
    (:Person:`__Entity__` {name: "John"})-[:CORRELATES_TO]->(:Person {person_name: "John"})


    Args:
        label (str): The label of the entity and domain nodes.
        entity_key (str): The key of the entity node.
        domain_key (str): The key of the domain node.
        similarity (float, optional): The similarity threshold for correlation. Defaults to 0.9.

    Returns:
        dict: A dictionary containing the correlation between the entity and domain nodes.
    """
    results = graphdb.query("""
    MATCH (entity:$($entityLabel):`__Entity__`),(domain:$($entityLabel))
    WHERE apoc.text.jaroWinklerDistance(entity[$entityKey], domain[$domainKey]) < $distance
    MERGE (entity)-[r:CORRESPONDS_TO]->(domain)
    ON CREATE SET r.created_at = datetime() // MERGE sub-clause when the relationship is newly created
    ON MATCH SET r.updated_at = datetime()  // MERGE sub-clause when the relationship already exists
    RETURN $entityLabel as entityLabel, count(r) as relationshipCount
    """, {
        "entityLabel": label,
        "entityKey": entity_key,
        "domainKey": domain_key,
        "distance": (1.0 - similarity)
    })

    if results['status'] == 'error':
        raise Exception(results['message'])

    return results['query_result']

if __name__ == "__main__":
    unique_entity_labels = find_unique_entity_labels()
    print(unique_entity_labels)

    # try out the function to get the unique keys for
    # subject nodes labeled as Product
    find_unique_entity_keys("Product")

    find_unique_domain_keys("Product")

    print(normalize_key("Product", "Product_name"))
    print(normalize_key("Product", "Product Name"))
    print(normalize_key("Product", "product name"))
    print(normalize_key("Product", "price"))

    label = "Product"
    entity_keys = find_unique_entity_keys(label)
    domain_keys = find_unique_domain_keys(label)

    # try correlating with a low-ish threshold
    correlated_keys = correlate_entity_and_domain_keys(label, entity_keys, domain_keys, similarity=0.5)

    print(f"{label} correlated keys (entity key, domain key, similarity score)...")

    # show the keys
    print(correlated_keys)

    correlate_subject_and_domain_nodes("Product", "name", "product_name")

    # do it all:
    # - loop over all entity labels
    # - correlate the keys
    # - correlate (and connect) the nodes
    for entity_label in find_unique_entity_labels():
        print(f"Correlating entities labeled {entity_label}...")

        entity_keys = find_unique_entity_keys(entity_label)
        domain_keys = find_unique_domain_keys(entity_label)

        correlated_keys = correlate_entity_and_domain_keys(entity_label, entity_keys, domain_keys, similarity=0.8)

        if (len(correlated_keys) > 0):
            top_correlated_keypair = correlated_keys[0]
            print("\tbased on:", top_correlated_keypair)
            correlate_subject_and_domain_nodes(entity_label, top_correlated_keypair[0], top_correlated_keypair[1])
        else:
            print("\tNo correlation found")