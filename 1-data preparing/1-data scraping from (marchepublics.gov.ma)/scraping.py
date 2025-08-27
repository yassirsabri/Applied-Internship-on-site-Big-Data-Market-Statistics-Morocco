import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import logging
import random
from datetime import datetime
import base64
import re

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarchesPublicsScraper:
    def __init__(self, headless=False):
        """
        Initialise le scraper
        
        Args:
            headless (bool): Mode sans interface graphique
        """
        self.driver = None
        self.wait = None
        self.setup_driver(headless)
    
    def setup_driver(self, headless=False):
        """Configure le driver Chrome avec les options nécessaires"""
        chrome_options = Options()
        
        if headless:
            chrome_options.add_argument('--headless')
        
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
        
        # Augmenter les timeouts
        chrome_options.add_argument('--timeout=120000')
        chrome_options.add_argument('--script-timeout=120000')
        chrome_options.add_argument('--page-load-strategy=normal')
        
        # Options pour éviter les popups
        chrome_options.add_argument('--disable-popup-blocking')
        chrome_options.add_argument('--disable-extensions')
        chrome_options.add_argument('--disable-plugins')
        chrome_options.add_argument('--disable-notifications')
        
        self.driver = webdriver.Chrome(options=chrome_options)
        
        # Configurer les timeouts
        self.driver.set_page_load_timeout(120)
        self.driver.set_script_timeout(120)
        self.wait = WebDriverWait(self.driver, 30)
        
        # Créer le dossier pour les PDFs
        os.makedirs('pdfs', exist_ok=True)
    
    def random_delay(self, min_delay=1, max_delay=3):
        """Ajoute un délai aléatoire pour éviter la détection"""
        delay = random.uniform(min_delay, max_delay)
        time.sleep(delay)
    
    def navigate_to_initial_page(self):
        """Navigue vers la page des marchés publics"""
        try:
            url = "https://www.marchespublics.gov.ma/index.php?page=entreprise.EntrepriseAdvancedSearch&AllAnn"
            logger.info(f"Navigation vers: {url}")
            self.driver.get(url)
            
            # Attendre que la page soit chargée
            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            self.random_delay(3, 5)
            
            logger.info("Page chargée avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de la navigation: {e}")
            raise
    
    def fill_search_form(self, start_date="01/01/2020", end_date=""):
        """Remplit le formulaire de recherche avec gestion améliorée des erreurs"""
        try:
            logger.info(f"Remplissage du formulaire avec date de début: {start_date}")
            
            # Attendre que le formulaire soit présent
            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "form")))
            
            # Recherche améliorée des champs de date
            date_start_field = None
            date_end_field = None
            
            # Méthodes multiples pour trouver les champs de date
            xpath_expressions = [
                "//input[@type='text' and contains(@name, 'date')]",
                "//input[@type='text' and contains(@id, 'date')]",
                "//td[contains(text(), 'Entre le')]/following-sibling::td//input[@type='text']",
                "//td[contains(text(), 'Date')]/following-sibling::td//input[@type='text']",
                "//input[@type='text'][position()<=10]"  # Prendre les 10 premiers champs text
            ]
            
            for xpath in xpath_expressions:
                try:
                    elements = self.driver.find_elements(By.XPATH, xpath)
                    if elements:
                        date_start_field = elements[0]
                        if len(elements) > 1:
                            date_end_field = elements[1]
                        break
                except:
                    continue
            
            # Remplir les champs de date si trouvés
            if date_start_field:
                try:
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", date_start_field)
                    self.random_delay(1, 2)
                    
                    date_start_field.clear()
                    self.random_delay(0.5, 1)
                    date_start_field.send_keys(start_date)
                    logger.info(f"Date de début saisie: {start_date}")
                    
                    if end_date and date_end_field:
                        self.random_delay(1, 2)
                        date_end_field.clear()
                        self.random_delay(0.5, 1)
                        date_end_field.send_keys(end_date)
                        logger.info(f"Date de fin saisie: {end_date}")
                    
                except Exception as e:
                    logger.error(f"Erreur lors de la saisie des dates: {e}")
            else:
                logger.warning("Aucun champ de date trouvé - continuer sans dates")
            
        except Exception as e:
            logger.error(f"Erreur lors du remplissage du formulaire: {e}")
    
    def launch_search(self):
        """Lance la recherche avec gestion robuste"""
        try:
            logger.info("Recherche du bouton 'Lancer la recherche'")
            
            self.random_delay(1, 2)
            
            # Méthodes multiples pour trouver le bouton
            button_selectors = [
                "//input[@value='Lancer la recherche']",
                "//input[contains(@value, 'Lancer')]",
                "//input[contains(@value, 'recherche')]",
                "//button[contains(text(), 'Lancer')]",
                "//button[contains(text(), 'recherche')]",
                "//input[@type='submit']",
                "//button[@type='submit']"
            ]
            
            search_button = None
            
            for selector in button_selectors:
                try:
                    button = self.driver.find_element(By.XPATH, selector)
                    if button and button.is_displayed():
                        search_button = button
                        break
                except:
                    continue
            
            if search_button:
                try:
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", search_button)
                    self.random_delay(1, 2)
                    
                    # Essayer clic normal puis JavaScript
                    try:
                        search_button.click()
                    except:
                        self.driver.execute_script("arguments[0].click();", search_button)
                    
                    logger.info("Bouton 'Lancer la recherche' cliqué avec succès")
                    
                    # Attendre le chargement de la page de résultats
                    self.random_delay(5, 8)
                    self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                    
                except Exception as e:
                    logger.error(f"Erreur lors du clic: {e}")
                    raise
            else:
                logger.error("Bouton 'Lancer la recherche' non trouvé")
                raise Exception("Bouton de recherche non trouvé")
                
        except Exception as e:
            logger.error(f"Erreur lors du lancement de la recherche: {e}")
            raise
    
    def set_results_per_page(self, results_per_page=500):
        """Configure le nombre de résultats par page"""
        try:
            logger.info(f"Configuration pour {results_per_page} résultats par page")
            
            # Chercher le dropdown de pagination
            select_elements = self.driver.find_elements(By.TAG_NAME, "select")
            
            for select_elem in select_elements:
                try:
                    options = select_elem.find_elements(By.TAG_NAME, "option")
                    
                    # Chercher l'option avec le bon nombre
                    for option in options:
                        if str(results_per_page) in option.text:
                            select = Select(select_elem)
                            select.select_by_visible_text(option.text)
                            logger.info(f"Sélectionné: {option.text}")
                            
                            self.random_delay(3, 5)
                            # Attendre rechargement
                            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                            return
                            
                except Exception as e:
                    continue
            
            logger.warning("Impossible de configurer le nombre de résultats par page")
            
        except Exception as e:
            logger.error(f"Erreur lors de la configuration: {e}")
    
    def save_page_as_pdf(self, page_number=1):
        """Sauvegarde la page en PDF uniquement"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_filename = f"pdfs/marches_publics_page_{page_number:03d}_{timestamp}.pdf"
            
            logger.info(f"Sauvegarde de la page {page_number} en PDF")
            
            # Attendre que la page soit complètement chargée
            self.random_delay(2, 3)
            
            # Utiliser Chrome DevTools Protocol pour générer le PDF
            try:
                result = self.driver.execute_cdp_cmd("Page.printToPDF", {
                    "format": "A4",
                    "landscape": False,
                    "printBackground": True,
                    "marginTop": 0.4,
                    "marginBottom": 0.4,
                    "marginLeft": 0.4,
                    "marginRight": 0.4,
                    "scale": 0.8,
                    "preferCSSPageSize": True
                })
                
                # Sauvegarder le PDF
                with open(pdf_filename, 'wb') as f:
                    f.write(base64.b64decode(result['data']))
                
                logger.info(f"PDF sauvegardé: {pdf_filename}")
                
            except Exception as pdf_error:
                logger.error(f"Erreur PDF: {pdf_error}")
                # Fallback: capture d'écran
                screenshot_filename = f"pdfs/marches_publics_page_{page_number:03d}_{timestamp}.png"
                self.driver.save_screenshot(screenshot_filename)
                logger.info(f"Capture d'écran sauvegardée: {screenshot_filename}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde page {page_number}: {e}")
    
    def get_current_page_info(self):
        """Récupère les informations sur la page actuelle"""
        try:
            # Chercher les informations de pagination
            page_info_selectors = [
                "//text()[contains(., 'Page')]",
                "//text()[contains(., 'page')]",
                "//*[contains(text(), 'Page')]",
                "//*[contains(text(), 'de')]",
                ".pagination",
                "[class*='page']"
            ]
            
            current_page = 1
            total_pages = 229  # Valeur par défaut basée sur votre indication
            
            for selector in page_info_selectors:
                try:
                    if selector.startswith("//text()"):
                        continue  # Skip text() selectors for now
                    
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if not elements:
                        elements = self.driver.find_elements(By.XPATH, selector)
                    
                    for element in elements:
                        text = element.text.strip()
                        if text:
                            # Extraire les numéros de page
                            numbers = re.findall(r'\d+', text)
                            if len(numbers) >= 2:
                                current_page = int(numbers[0])
                                total_pages = int(numbers[-1])
                                return current_page, total_pages
                            elif "Page" in text and len(numbers) >= 1:
                                current_page = int(numbers[0])
                                
                except Exception as e:
                    continue
            
            return current_page, total_pages
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des infos de page: {e}")
            return 1, 229
    
    def navigate_to_page(self, page_number):
        """Navigue vers une page spécifique en utilisant le champ de saisie"""
        try:
            logger.info(f"Navigation vers la page {page_number}")
            
            # Attendre que la page soit stable
            self.random_delay(2, 3)
            
            # Chercher le champ de saisie de numéro de page
            page_input_selectors = [
                "//input[@type='text' and @value]",  # Champ avec une valeur
                "//input[@type='text'][contains(@name, 'page')]",
                "//input[@type='text'][contains(@id, 'page')]",
                "//input[@type='text'][string-length(@value) <= 3]",  # Champ avec valeur courte (numéro de page)
                "//td[contains(text(), 'page')]/following-sibling::td//input[@type='text']",
                "//input[@type='text'][last()]"  # Dernier champ input text
            ]
            
            page_input = None
            
            for selector in page_input_selectors:
                try:
                    elements = self.driver.find_elements(By.XPATH, selector)
                    for element in elements:
                        if element.is_displayed() and element.is_enabled():
                            # Vérifier si c'est vraiment le champ de page
                            value = element.get_attribute('value')
                            if value and value.isdigit() and int(value) <= 229:
                                page_input = element
                                break
                    if page_input:
                        break
                except:
                    continue
            
            if page_input:
                try:
                    # Scroller vers le champ
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", page_input)
                    self.random_delay(1, 2)
                    
                    # Vider le champ et saisir le nouveau numéro
                    page_input.clear()
                    self.random_delay(0.5, 1)
                    page_input.send_keys(str(page_number))
                    
                    logger.info(f"Numéro de page {page_number} saisi")
                    
                    # Appuyer sur Entrée pour valider
                    from selenium.webdriver.common.keys import Keys
                    page_input.send_keys(Keys.RETURN)
                    
                    logger.info("Validation avec Entrée")
                    
                    # Attendre le chargement de la nouvelle page
                    self.random_delay(5, 8)
                    self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                    
                    return True
                    
                except Exception as e:
                    logger.error(f"Erreur lors de la saisie du numéro de page: {e}")
                    return False
            else:
                logger.warning("Champ de saisie de page non trouvé - essayer méthode alternative")
                return self.navigate_to_next_page_alternative()
                
        except Exception as e:
            logger.error(f"Erreur lors de la navigation vers la page {page_number}: {e}")
            return False
    
    def navigate_to_next_page_alternative(self):
        """Méthode alternative: clic sur le bouton suivant"""
        try:
            logger.info("Utilisation de la méthode alternative (bouton suivant)")
            
            # Chercher le bouton suivant
            next_selectors = [
                "//a[contains(text(), '»')]",
                "//a[contains(text(), '>')]",
                "//a[contains(@title, 'Page suivante')]",
                "//a[contains(@title, 'suivant')]",
                "//input[@value='»']",
                "//input[@value='>']"
            ]
            
            for selector in next_selectors:
                try:
                    elements = self.driver.find_elements(By.XPATH, selector)
                    for element in elements:
                        if element.is_displayed() and element.is_enabled():
                            # Cliquer sur le bouton
                            self.driver.execute_script("arguments[0].scrollIntoView(true);", element)
                            self.random_delay(1, 2)
                            
                            try:
                                element.click()
                            except:
                                self.driver.execute_script("arguments[0].click();", element)
                            
                            logger.info("Clic sur bouton suivant réussi")
                            
                            # Attendre le chargement
                            self.random_delay(5, 8)
                            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                            
                            return True
                except:
                    continue
            
            logger.warning("Aucune méthode de navigation trouvée")
            return False
            
        except Exception as e:
            logger.error(f"Erreur dans la méthode alternative: {e}")
            return False
    
    def scrape_all_pages(self, start_date="01/01/2020", end_date="", max_pages=250):
        """Scrape toutes les pages avec navigation par numéro de page"""
        try:
            logger.info("=== DÉBUT DU SCRAPING ===")
            
            # Étape 1: Navigation et recherche initiale
            self.navigate_to_initial_page()
            self.fill_search_form(start_date, end_date)
            self.launch_search()
            
            # Étape 2: Configuration
            self.set_results_per_page(500)
            
            # Étape 3: Scraping des pages
            current_page = 1
            consecutive_failures = 0
            max_consecutive_failures = 3
            
            while current_page <= max_pages:
                try:
                    logger.info(f"=== TRAITEMENT DE LA PAGE {current_page} ===")
                    
                    # Sauvegarder la page actuelle
                    self.save_page_as_pdf(current_page)
                    
                    # Vérifier si on est à la dernière page
                    if current_page >= 229:  # Nombre total de pages connu
                        logger.info("Dernière page atteinte (229)")
                        break
                    
                    # Naviguer vers la page suivante
                    next_page = current_page + 1
                    if self.navigate_to_page(next_page):
                        consecutive_failures = 0
                        current_page = next_page
                    else:
                        consecutive_failures += 1
                        logger.warning(f"Échec navigation vers page {next_page} - tentative {consecutive_failures}/{max_consecutive_failures}")
                        
                        if consecutive_failures >= max_consecutive_failures:
                            logger.error("Trop d'échecs consécutifs - arrêt du scraping")
                            break
                        
                        # Attendre plus longtemps avant de réessayer
                        self.random_delay(10, 15)
                        current_page += 1
                
                except Exception as e:
                    logger.error(f"Erreur sur la page {current_page}: {e}")
                    consecutive_failures += 1
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error("Trop d'erreurs consécutives - arrêt")
                        break
                    
                    # Attendre et continuer
                    self.random_delay(5, 10)
                    current_page += 1
            
            logger.info(f"=== SCRAPING TERMINÉ - {current_page} pages traitées ===")
            
        except Exception as e:
            logger.error(f"Erreur critique dans scrape_all_pages: {e}")
            raise
    
    def close(self):
        """Ferme le navigateur"""
        if self.driver:
            self.driver.quit()
            logger.info("Navigateur fermé")


def main():
    """Fonction principale"""
    scraper = None
    try:
        logger.info("=== DÉMARRAGE DU SCRAPER MARCHÉS PUBLICS ===")
        
        # Créer le scraper (headless=True pour mode sans interface)
        scraper = MarchesPublicsScraper(headless=False)
        
        # Lancer le scraping avec paramètres
        scraper.scrape_all_pages(
            start_date="01/01/2020", 
            end_date="",
            max_pages=250  # Augmenté pour couvrir les 229 pages
        )
        
        logger.info("=== SCRAPING TERMINÉ AVEC SUCCÈS ===")
        
    except Exception as e:
        logger.error(f"Erreur dans le programme principal: {e}")
        
    finally:
        # Fermer le navigateur
        if scraper:
            scraper.close()


if __name__ == "__main__":
    main()
