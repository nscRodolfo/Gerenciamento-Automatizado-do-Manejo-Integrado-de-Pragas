package my.aplication.manejointeligentedepragas;

import android.content.Intent;
import android.content.SharedPreferences;

//import android.support.annotation.NonNull;
import androidx.annotation.NonNull;
//import android.support.design.widget.NavigationView;
import com.google.android.material.navigation.NavigationView;
//import android.support.v4.view.GravityCompat;
import androidx.core.view.GravityCompat;
//import android.support.v4.widget.DrawerLayout;
import androidx.drawerlayout.widget.DrawerLayout;
//import android.support.v7.app.ActionBarDrawerToggle;
import androidx.appcompat.app.ActionBarDrawerToggle;
//import android.support.v7.app.AppCompatActivity;
import androidx.appcompat.app.AppCompatActivity;
//import android.support.v7.widget.Toolbar;
import androidx.appcompat.widget.Toolbar;

import android.os.Bundle;
import android.view.MenuItem;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import my.aplication.manejointeligentedepragas.Crontroller.Controller_Usuario;

import com.example.manejointeligentedepragas.R;

public class SobreMIP extends AppCompatActivity implements NavigationView.OnNavigationItemSelectedListener
{

    TextView pergunta1;
    TextView resposta1;
    TextView adicional;
    TextView adicional1;
    TextView adicional2;
    TextView adicional3;
    TextView adicional4;
    TextView adicional5;
    TextView pergunta2;
    TextView resposta2;
    TextView pergunta3;
    TextView resposta3;
    TextView pergunta4;
    TextView resposta4;
    TextView pergunta5;
    TextView resposta5;
    TextView pergunta6;
    TextView resposta6;
    TextView pergunta7;
    TextView resposta7;
    TextView pergunta8;
    TextView resposta8;
    TextView pergunta9;
    TextView resposta9;

    String tipoUsu;

    private DrawerLayout drawerLayout;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_sobre_mip);



        //menu novo
        Toolbar toolbar = findViewById(R.id.toolbar_sobre_mip);
        setSupportActionBar(toolbar);
        drawerLayout= findViewById(R.id.drawer_layout_sobre_mip);
        NavigationView navigationView = findViewById(R.id.nav_view_sobre_mip);
        navigationView.setNavigationItemSelectedListener(this);
        ActionBarDrawerToggle toggle = new ActionBarDrawerToggle(this, drawerLayout, toolbar, R.string.navigation_drawer_open, R.string.navigation_drawer_close);
        drawerLayout.addDrawerListener(toggle);
        toggle.syncState();
        View headerView = navigationView.getHeaderView(0);

        Controller_Usuario controller_usuario = new Controller_Usuario(getBaseContext());
        String nomeUsu = controller_usuario.getUser().getNome();
        String emailUsu = controller_usuario.getUser().getEmail();

        TextView nomeMenu = headerView.findViewById(R.id.nomeMenu);
        nomeMenu.setText(nomeUsu);

        TextView emailMenu = headerView.findViewById(R.id.emailMenu);
        emailMenu.setText(emailUsu);

        setTitle("MIP² | Sobre o MIP");


        pergunta1 = findViewById(R.id.tvPergunta1);
        pergunta1.setText("O que é o Manejo Integrado de Pragas (MIP)?");
        resposta1 = findViewById(R.id.tvResp1);
        resposta1.setText("O MIP, como é conhecido o Manejo Integrado de Pragas, trata-se de uma filosofia de controle de pragas que procura preservar e incrementar os fatores de mortalidade natural, através do uso integrado de todas as técnicas de combate possíveis, selecionadas com base nos parâmetros econômicos, ecológicos e sociológicos. O MIP possui 4 componentes principais sendo eles: diagnose/identificação das pragas; amostragem das populações de pragas; tomada de decisão e a seleção dos métodos de controle.\nA implantação destes componentes no manejo das pragas garante que a adoção dos métodos de controle só será realizada quando a praga possuir uma densidade populacional que afete economicamente o cultivo. Com isso, a divisão de uma área extensa, em talhões menores e com um número fixo de plantas a serem amostradas, assegura a representatividade do cultivo. Assim o método de controle só deverá ser realizado no talhão que após uma amostragem fique determinado que a população de pragas está ocasionando danos significativos.");

        adicional = findViewById(R.id.tvAdicional);
        adicional.setText("Como realizar a identificação correta das pragas?");
        adicional1 = findViewById(R.id.tvAdicional1);
        adicional1.setText("Para realizar a identificação correta das pragas em sua propriedade você deverá avaliar as pragas-chave que acometem a cultura escolhida. Utilize as fotos de cada praga para comparar com as pragas encontradas em campo. Na descrição de cada praga é possível encontrar o local, horário e estágio fenológico em que a praga atua sobre as plantas, bem como o material necessário para realizar sua identificação e os danos por elas causados. É importante identificar também os inimigos naturais dessas pragas, para evitar eliminá-los da área de cultivo. O plantio de espécies atrativas é recomendado para aumentar o controle natural realizado pela predação e demais relações entre os inimigos naturais e as pragas-chaves de sua cultura.");

        adicional2 = findViewById(R.id.tvAdicional2);
        adicional2.setText("Como realizar a amostragem?");
        adicional3 = findViewById(R.id.tvAdicional3);
        adicional3.setText("A amostragem pode ser realizada para determinar se a praga se encontra no nível de dano econômico (onde necessita fazer o uso de um método de controle). Para realiza-la é necessário seguir alguns passos importantes, realizando a inspeção das amostras de todas as plantas solicitas para cada talhão, desta forma estará assegurado que a tomada de decisão será correta, ou seja, a contagem foi representativa - caso tenha menos plantas que o necessário, realize o plano de amostragem apenas com o número de plantas presentes em sua cultura. \n" +
                "Divisão da área em talhões: para realizar a amostragem é necessário dividir a área de cultivo em talhões, cada talhão possui um tamanho máximo em função da cultura a ser amostrada, listado na cultura escolhida. O talhão trata-se de uma área que contém as plantas com o mesmo genótipo, idade, espaçamento, sistema de condução, tipo de solo e topografia.\n" +
                "Quando você insere o tamanho da cultura, temos um valor de referência que limita o tamanho do talhão para cada cultura, desta forma, ao inserir uma área maior do que o limite, o aplicativo automaticamente cria outros talhões, assim evitamos erros na tomada de decisão. No entanto, algumas propriedades podem apresentar mais de um talhão - em função de suas características – e ainda assim não extrapolar o tamanho máximo para cada talhão, o que não configura a necessidade do aplicativo de criar outros talhões automaticamente. Afim de evitar uma amostragem incorreta (sem levar em consideração a diferença de cada talhão da propriedade), neste caso é necessário adicionar outros talhões na cultura de forma manual. " +
                "Caso o aplicativo defina os talhões de forma automática, verifique nas informações da cultura o tamanho máximo de cada talhão, para que você possa criar um mapa da propriedade contendo os talhões. O mesmo deve ser feito no caso da inserção manual dos talhões, isso facilita o planejamento administrativo da propriedade e a realização da amostragem.\n" +
                "Importante: em todo sistema de produção é recomendado ter algumas planilhas ou documentos para auxiliar na gestão no manejo como na coleta de material para análise química do solo e foliar, a operacionalização dos tratos culturais, aplicação de métodos de controle ou adubação e o escalonamento da colheita. Neste documento cada talhão deve possuir algumas informações como: número ou nome do talhão (o mesmo que no aplicativo), tamanho (número de plantas), cultivar, data de plantio, localização entre outras. Assim ao final do monitoramento deste talhão, por meios dos relatórios você poderá preencher com as informações disponíveis como: data da avaliação, responsável pela amostragem, decisão alcançada por meio do relatório de métodos de controle utilizados, entre outros ou fazer o download nos relatórios.\n" +
                " \n" +
                "Caminhamento: a forma de caminhamento para realizar esta amostragem é sugerida na tela de amostragem, contendo o número de pontos e plantas por talhão, bem como o número fixo de plantas a serem amostrados. No entanto é possível fazer o caminhamento de várias formas, devendo realiza-la da forma mais representativa em cada talhão. Alguns exemplos de caminhamento são: U, X, C, Z e por pontos.\n" +
                "\n" +
                "Amostras: as amostras no MIP representam a unidade de avaliação da praga ou inimigo natura, no MIP² as amostras selecionadas são feitas para avaliar apenas as pragas, cada amostra está listada nas informações das pragas ou no ícone de informação na tela de amostragem. As formas de obtenção das amostras também estão listadas em cada praga, sendo a mais comum a contagem direta da população do inseto com o uso de lupa. O número de amostras por talhão nos planos convencionais de amostragem é fixo, portanto cada talhão no MIP² terá um valor fixo de plantas a serem amostrados, listados também na tela de amostragem. A época e frequência de amostragem é importante para determinar se as pragas se encontram sob controle e se o método de controle teve resultado. A amostragem deve ser realizada com maior frequência em períodos de maiores incidências das pragas e de maior suscetibilidade da cultura, listados para cada praga.");

        adicional4 = findViewById(R.id.tvAdicional4);
        adicional4.setText("Como é feita a tomada de decisão e a seleção de métodos de controle?");
        adicional5 = findViewById(R.id.tvAdicional5);
        adicional5.setText("De acordo com a amostragem realizada, o aplicativo irá informar se há ou não a necessidade de realizar o controle da praga. É importante lembrar que a seleção do método de controle só deverá ser realizada para controle de uma praga por vez na mesma cultura, não podendo aplicar mais caldas/extratos durante o período descrito em cada método de controle, a fim de evitar fitotoxidez na planta. Portanto, se a cultura apresenta mais de uma praga acima do nível de dano econômico (no qual precisa de controle) é importante verificar a calda/extrato que melhor se adeque para ambos os tratamentos.\n" +
                " Os métodos de controle listados neste aplicativo são retirados de cartilhas da EMBRAPA, MAPA e demais órgãos e grupos competentes, bem como trabalhos científicos. Por se tratarem de métodos alternativos, estes deverão ser testados na localidade do agricultor, sob suas características climáticas para verificar se há ou não a eficácia. Cada método de controle possui uma especificação quanto a fabricação e efeitos nocivos, devendo produzi-lo com atenção e de forma responsável.");




        pergunta2 = findViewById(R.id.tvPergunta2);
        pergunta2.setText("O que é o Monitoramento Inteligente de Pragas (MIP²)? ");
        resposta2 = findViewById(R.id.tvResp2);
        resposta2.setText("O MIP² trata-se de uma ferramenta computacional agroecológica baseada no MIP, que tem a função de facilitar a adoção do Manejo Integrado de Pragas nas propriedades com cultivo orgânico/agroecológico.");

        pergunta3 = findViewById(R.id.tvPergunta3);
        pergunta3.setText("Como utilizar o aplicativo?");
        resposta3 = findViewById(R.id.tvResp3);
        resposta3.setText("Para utilizar o aplicativo é necessário fazer um cadastro preenchendo as informações requisitadas, selecionando qual o tipo de usuário. O usuário tipo PROPRIETÁRIO tem a acesso total as funções oferecidas pelo MIP² referente a propriedade como: adicionar/editar/excluir propriedade, adicionar/excluir culturas e adicionar/excluir FUNCIONARIO.\n" +
                "O FUNCIONARIO tem função limitada no aplicativo, não podendo gerir a propriedade ou adicionar culturas, no entanto ainda tem acesso as informações para que seja possível executar o plano de amostragem, como ver informações das culturas, pragas e métodos de controle, adicionar praga, selecionar o método de controle adotado e gerar relatórios. ");


        pergunta4 = findViewById(R.id.tvPergunta4);
        pergunta4.setText("Como gerir minhas propriedades?");
        resposta4 = findViewById(R.id.tvResp4);
        resposta4.setText("Após entrar no sistema, você (proprietário) será direcionado para a tela de propriedades, onde é possível adicionar propriedades e ver quais propriedades estão cadastradas no sistema. Ao selecionar uma propriedade você será levado para uma tela onde é possível ver quantos funcionários estão vinculados à sua propriedade e quais culturas estão presentes nessa propriedade. Para vincular funcionários, clique no cartão FUNCIONÁRIOS > ADICIONAR FUNCIONÁRIOS e peça à pessoa que insira suas credenciais de entrada. Ao vincular um funcionário à sua propriedade, ele poderá realizar ações nas culturas existentes na propriedade.");


        pergunta5 = findViewById(R.id.tvPergunta5);
        pergunta5.setText("Como faço para adicionar uma cultura a propriedade?");
        resposta5 = findViewById(R.id.tvResp5);
        resposta5.setText("Após adicionar a propriedade você (proprietário) poderá escolher dentre as culturas cadastradas no aplicativo, devendo preencher o tamanho da cultura em hectares para que o aplicativo defina quantos talhões serão necessários para realizar o plano de amostragem. Também é possível nomear cada talhão da propriedade de acordo com a necessidade do produtor.");


        pergunta6 = findViewById(R.id.tvPergunta6);
        pergunta6.setText("Como identificar e adicionar uma praga?");
        resposta6 = findViewById(R.id.tvResp6);
        resposta6.setText("Para identificar uma praga em uma cultura o usuário deve selecionar no menu suspenso a cultura na qual se deseja inspecionar, ao final da descrição da cultura você encontra a opção PRAGAS-CHAVE e poderá comparar as fotos e as descrições com a praga encontrada em campo.\n" +
                "Para adicionar uma praga que identificou em campo ao seu cultivo - para fins monitoramento e controle - selecione o talhão da cultura, clique em PRAGAS ATUANTES> ADICIONAR PRAGAS> selecione a praga identificada e clique em SALVAR. As pragas adicionadas terão diferentes cores, indicando diferentes situações que podem ser descritas ao clicar no botão de informação (i). \n" +
                "Obs: O usuário contará também com a opção INFORMAÇÕES DA PRAGA na tela de seleção da praga, basta selecionar a praga e clicar nesta opção. É provável que cada talhão de uma cultura apresente uma praga-chave diferente em função das características e do manejo do talhão, portanto, deve-se identificar o talhão em campo para evitar erros na amostragem. ");


        pergunta7 = findViewById(R.id.tvPergunta7);
        pergunta7.setText("Como realizar o plano de amostragem?");
        resposta7 = findViewById(R.id.tvResp7);
        resposta7.setText("Ao selecionar a cultura onde se localiza a praga a ser amostrada clique em REALIZAR PLANO DE AMOSTRAGEM e selecione a praga. Siga as informações descritas na tela inspecionando o número de plantas exigido, caso tenha dúvida sobre a localização da praga na planta (AMOSTRA), clique no botão (i). Você deverá inspecionar o número de plantas descrito para cada talhão, caso tenha menos plantas que o necessário, realize o plano de amostragem apenas com o número de plantas presentes em sua cultura. Para realizar a contagem clique no botão ENCONTRADO ou NÃO ENCONTRADO de acordo com a AMOSTRA para cada planta até completar o plano, se cometer um erro, é possível clicar em CORRIGIR, corrigindo a última planta amostrada, podendo também segurar o botão para recomeçar a contagem, caso tenha cometido erros consecutivos. Ao FINALIZAR a contagem você será levado para uma tela que vai informar se há necessidade de aplicar algum método de controle. Havendo necessidade, você poderá selecionar o método de controle, devendo aplicá-lo assim que possível. \n" +
                "ATENÇÃO: Ao selecionar APLICAR o sistema entende que você realizou a aplicação do método, sendo necessário aguardar o tempo exigido, que ficará descrito no cartão da sua cultura. Caso selecione APLICAR DEPOIS, a sua praga ficará vermelha na tela de pragas atuantes, o que indica ser necessária uma aplicação para o controle desta.");


        pergunta8 = findViewById(R.id.tvPergunta8);
        pergunta8.setText("Como realizar a aplicação do método de controle?");
        resposta8 = findViewById(R.id.tvResp8);
        resposta8.setText("É indicado que, ao detectar a necessidade de uma aplicação, o usuário realize-a o quanto antes, aplicando sobre TODA a cultura e NÃO apenas sobre as plantas infestadas. Para mais informações leia a descrição do método a ser utilizado.");


        pergunta9 = findViewById(R.id.tvPergunta9);
        pergunta9.setText("Como ver o resultado da aplicação do MIP²?");
        resposta9 = findViewById(R.id.tvResp9);
        resposta9.setText("Em cada cultura é possível clicar no botão GERAR RELATÓRIOS, onde são descritos os relatórios de maior relevância para a gestão de sua propriedade.\n");
    }

    @Override
    public void onBackPressed() {
        if(drawerLayout.isDrawerOpen(GravityCompat.START)){
            drawerLayout.closeDrawer(GravityCompat.START);
        }else {
            super.onBackPressed();
        }
    }

    @Override
    public boolean onNavigationItemSelected(@NonNull MenuItem menuItem) {
        switch (menuItem.getItemId()){
            case R.id.drawerPerfil:
                Controller_Usuario cu = new Controller_Usuario(getBaseContext());
                tipoUsu = cu.getUser().getTipo();
                if(tipoUsu == null){
                    Toast.makeText(SobreMIP.this,"Para acessar seu perfil, faça login!", Toast.LENGTH_LONG).show();
                }else{
                    Intent i= new Intent(this, Perfil.class);
                    startActivity(i);
                }
                break;
            case R.id.drawerProp:
                Controller_Usuario cu1 = new Controller_Usuario(getBaseContext());
                tipoUsu = cu1.getUser().getTipo();
                if(tipoUsu==null){
                    Toast.makeText(SobreMIP.this,"Para acessar as propriedades, faça login!", Toast.LENGTH_LONG).show();
                }else{
                    Intent prop= new Intent(this, Propriedades.class);
                    startActivity(prop);
                }
                break;

            case R.id.drawerPlantas:
                Intent j = new Intent(this, VisualizaPlantas.class);
                startActivity(j);
                break;

            case R.id.drawerPrag:
                Intent k = new Intent(this, VisualizaPragas.class);
                startActivity(k);
                break;

            case R.id.drawerMet:
                Intent l = new Intent(this, VisualizaMetodos.class);
                startActivity(l);
                break;

            case R.id.drawerSobreMip:
                Intent p = new Intent(this, SobreMIP.class);
                startActivity(p);
                break;

            case R.id.drawerTutorial:
                SharedPreferences pref = getApplicationContext().getSharedPreferences("myPrefs",MODE_PRIVATE);
                SharedPreferences.Editor editor = pref.edit();
                editor.putBoolean("isIntroOpened",false);
                editor.commit();

                Intent intro = new Intent(this, IntroActivity.class);
                startActivity(intro);
                break;

            case R.id.drawerSobre:
                Intent pp = new Intent(this, Sobre.class);
                startActivity(pp);
                break;

            case R.id.drawerReferencias:
                Intent pi = new Intent(this, Referencias.class);
                startActivity(pi);
                break;

            case R.id.drawerRecomendações:
                Intent pa = new Intent(this, RecomendacoesMAPA.class);
                startActivity(pa);
                break;
        }
        drawerLayout.closeDrawer(GravityCompat.START);
        return true;
    }

}
